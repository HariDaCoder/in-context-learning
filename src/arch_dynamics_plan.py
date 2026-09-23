"""Isolated v3 plans. Family labels are excluded from training identity."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('arch_stageA_joint_scale', 'arch_stageB_head_sweep',
          'arch_stageC_depth_sweep', 'arch_stageD_selected_replicates',
          'arch_mechanism', 'arch_interaction_optional',
          'arch_parameter_matched_depth', 'arch_width_optional')
BLOCKED = 'ARCHITECTURE_BO_BLOCKED: no verified benign-overfitting region under the current protocol.'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_settings(path=None):
    defaults = json.loads((ROOT / 'src/conf/arch_dynamics.json').read_text())
    if path:
        defaults.update(json.loads(Path(path).read_text()))
    c = defaults
    for key in ('d', 'context', 'batch_size', 'train_steps', 'save_every', 'n_eval',
                'eval_batch_size', 'n_queries', 'probe_train_tasks', 'probe_test_tasks',
                'stable_window', 'bootstrap_samples'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if c['probe_test_tasks'] < 2 or not c['eval_seeds'] or c['rho_e'] != 0:
        raise ValueError('Independent noise and nonempty held-out evaluation are required')
    if any(r not in (0., .6, .9) for r in c['rhos']) or not c['rhos']:
        raise ValueError('Only IID, .9 and optional .6 supported')
    if any(type(s) is not int or s < 0 for s in c['diagnostic_steps']):
        raise ValueError('Diagnostic steps must be nonnegative integers')
    if c['precision'] not in ('float32', 'bfloat16', 'float16'):
        raise ValueError('Invalid precision')
    if not (0 < c['event_frequency'] <= 1 and 0 <= c['tau_probe'] <= 1):
        raise ValueError('Invalid event/probe thresholds')
    if c['learning_rate'] <= 0 or c['tau_fit'] < 0 or c['tau_gen'] < 0:
        raise ValueError('Invalid learning rate or BO thresholds')
    if c['harmful_threshold'] <= c['tau_gen']:
        raise ValueError('Harmful threshold must exceed generalization threshold')
    c['diagnostic_steps'] = sorted(set([0, c['train_steps']] +
                                      [s for s in c['diagnostic_steps'] if s <= c['train_steps']]))
    return c


def families():
    def spec(w, l, h, label):
        return dict(n_embd=w, n_layer=l, n_head=h, head_dim=w//h, architecture_label=label)
    return {
        'joint_scale': [spec(64,3,2,'Tiny'), spec(128,6,4,'Small'), spec(256,12,8,'Standard')],
        'head_sweep_fixed_width': [spec(256,12,h,f'H={h}') for h in (1,2,4,8,16)],
        'depth_sweep_fixed_width': [spec(256,l,8,f'L={l}') for l in (1,3,6,12,24)],
        'width_only': [spec(w,6,4,f'W={w}') for w in (64,128,256)],
        'interaction': [spec(256,l,h,f'H={h}, L={l}') for h in (2,8,16) for l in (3,12,24)],
    }


def select_conditions(audit_path, c, debug_snrs=None):
    """Conservative per-regime matching; old audits without coordinates cannot pass."""
    if debug_snrs:
        if len(debug_snrs) != 2 or any(not math.isfinite(s) or s <= 0 for s in debug_snrs):
            raise ValueError('Debug override requires two positive SNRs')
        return [dict(rho=r, snr=s, condition=f'exploratory_{i}', exploratory=True)
                for r in c['rhos'] for i,s in enumerate(debug_snrs)]
    if not Path(audit_path).is_file():
        raise RuntimeError(BLOCKED + ' Audit is missing.')
    report = json.loads(Path(audit_path).read_text())
    selected = []
    for rho in c['rhos']:
        rows = [r for r in report.get('rows', []) if r.get('fully_matched') is True
                and r.get('rho_x') == rho and r.get('rho_e') == 0
                and r.get('d') == c['d'] and r.get('k') == c['context']
                and isinstance(r.get('snr'), (float,int)) and math.isfinite(r['snr']) and r['snr'] > 0]
        def num(r, k, default=float('inf')):
            v = r.get(k)
            return v if isinstance(v, (int,float)) and math.isfinite(v) else default
        benign = [r for r in rows if
                  (num(r,'direct_bo_frequency',0) >= c['event_frequency']
                   and num(r,'duplicate_fit_ratio') <= c['tau_fit']
                   and num(r,'clean_gen_ratio') <= c['tau_gen']) or
                  (num(r,'linear_bo_frequency',0) >= c['event_frequency']
                   and num(r,'heldout_probe_r2',0) >= c['tau_probe']
                   and num(r,'linear_fit_ratio') <= c['tau_fit']
                   and num(r,'linear_clean_gen_ratio') <= c['tau_gen'])]
        harmful = [r for r in rows if num(r,'duplicate_fit_ratio') <= c['tau_fit']
                   and math.isfinite(num(r,'clean_gen_ratio'))
                   and num(r,'clean_gen_ratio') > c['harmful_threshold']]
        if not benign:
            raise RuntimeError(BLOCKED + f' rho={rho}, d={c["d"]}, k={c["context"]}.')
        if not harmful:
            raise RuntimeError('ARCHITECTURE_BO_BLOCKED: no verified harmful point for rho=' + str(rho))
        for name, candidates in [('verified_benign_snr',benign), ('verified_harmful_snr',harmful)]:
            row = sorted(candidates, key=lambda r: (r['snr'], str(r.get('checkpoint_id'))))[0]
            selected.append(dict(rho=rho, snr=row['snr'], condition=name,
                                 exploratory=False, evidence=row))
    return selected


def architecture_table(c, parameter_match=False):
    import torch
    from bo_architecture import ArchitectureSpec, instantiate_and_count, parameter_matched_depth_specs
    table = []
    # Preserve CPU RNG; counting does not change initialization/data streams.
    with torch.random.fork_rng(devices=[]):
        for family, specs in families().items():
            for spec in specs:
                count = instantiate_and_count(ArchitectureSpec(spec['n_embd'],spec['n_layer'],spec['n_head']),
                                              c['d'], c['context']+1, trainable_only=False)
                table.append(dict(spec, architecture_family=family, exact_parameter_count=count))
        standard = next(r['exact_parameter_count'] for r in table if r['architecture_label']=='Standard')
        if parameter_match:
            matches = parameter_matched_depth_specs(ArchitectureSpec(256,12,8), (3,6,12,24),
                        range(64,769,8), c['d'], c['context']+1, trainable_only=False)
            for m in matches:
                s=m.candidate
                table.append(dict(n_embd=s.n_embd,n_layer=s.n_layer,n_head=s.n_head,head_dim=s.head_dim,
                    architecture_family='parameter_matched_depth',architecture_label=f'L={s.n_layer}',
                    exact_parameter_count=m.parameter_count))
    for row in table:
        row['parameter_count_ratio_to_standard']=row['exact_parameter_count']/standard
        row['relative_parameter_error']=abs(row['exact_parameter_count']-standard)/standard
    assert len({r['exact_parameter_count'] for r in table if r['architecture_family']=='head_sweep_fixed_width'})==1
    return table


def plan(group, c, audit_path, table, debug_snrs=None, selection=None):
    conditions = select_conditions(audit_path,c,debug_snrs)
    group_family = dict(zip(GROUPS[:3], ('joint_scale','head_sweep_fixed_width','depth_sweep_fixed_width')))
    group_family.update(arch_width_optional='width_only',arch_interaction_optional='interaction',
                        arch_parameter_matched_depth='parameter_matched_depth')
    if group in ('arch_stageD_selected_replicates','arch_mechanism'):
        if not selection:
            raise ValueError('This stage requires --architecture-selection with explicitly selected shapes')
        chosen=json.loads(Path(selection).read_text())['selected_architectures']
        specs=[r for r in table if [r['n_embd'],r['n_layer'],r['n_head']] in chosen]
        seeds=range(5) if group=='arch_stageD_selected_replicates' else (0,)
    else:
        specs=[r for r in table if r['architecture_family']==group_family[group]]
        seeds=(0,)
    if group=='arch_interaction_optional':
        if debug_snrs:
            conditions=conditions[::2]
        else:
            # Observed point nearest the target probability; never interpolate an SNR.
            rows=json.loads(Path(audit_path).read_text()).get('rows',[])
            conditions=[]
            for rho in c['rhos']:
                candidates=[r for r in rows if r.get('fully_matched') and r.get('rho_x')==rho
                    and r.get('rho_e')==0 and r.get('d')==c['d'] and r.get('k')==c['context']
                    and isinstance(r.get('direct_bo_frequency'),(int,float)) and r.get('snr',0)>0]
                if not candidates: raise RuntimeError('No sampled near-boundary condition')
                row=min(candidates,key=lambda r:abs(r['direct_bo_frequency']-c['event_frequency']))
                conditions.append(dict(rho=rho,snr=row['snr'],condition='near_boundary_snr',
                                       exploratory=False,evidence=row))
    runs={}
    for spec in specs:
        for condition in conditions:
            for seed in seeds:
                model={k:spec[k] for k in ('n_embd','n_layer','n_head')}
                identity=dict(settings=c,model=model,rho=condition['rho'],snr=condition['snr'],
                              train_seed=seed,exploratory=condition['exploratory'],protocol_version=1)
                id_='arch_'+digest(identity)
                membership={k:spec[k] for k in ('architecture_family','architecture_label')}
                if id_ not in runs:
                    runs[id_]=dict(experiment_id=id_,**identity,condition=condition['condition'],
                        audit_evidence=condition.get('evidence'),memberships=[],
                        exact_parameter_count=spec['exact_parameter_count'])
                if membership not in runs[id_]['memberships']: runs[id_]['memberships'].append(membership)
    if not runs: raise ValueError('No architectures selected')
    return dict(group=group,settings=c,conditions=conditions,experiments=list(runs.values()),
                experiment_count=len(runs),audit_path=str(audit_path),architecture_table=table)
