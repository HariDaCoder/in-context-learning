"""Training-step figures, phase trajectories, and observed event summaries."""
import json
from collections import defaultdict
import numpy as np
from arch_dynamics_plan import write_json
from arch_dynamics_diagnostics import events


def mean(record,key):
    if key in record['metrics']: return record['metrics'][key]['mean']
    mech=record.get('mechanism')
    if not mech: return None
    rows=mech['heads'] if key in mech['heads'][0] else mech['layers']
    values=[r[key]['mean'] for r in rows if key in r and r[key]['mean'] is not None]
    return float(np.mean(values)) if values else None


def hierarchy_ci(runs,key,index,repeats=500):
    """Resample independent training runs, then evaluation seeds within each run.

    Each time index is handled separately; diagnostic times are never samples.
    """
    rng=np.random.default_rng(0); estimates=[]
    for _ in range(repeats):
        values=[]
        for j in rng.integers(len(runs),size=len(runs)):
            record=runs[j][index]
            seeds=[r['metrics'][key]['mean'] for r in record['evaluation_seeds']
                   if r['metrics'].get(key,{}).get('mean') is not None]
            if seeds: values.append(np.mean(rng.choice(seeds,len(seeds))))
        if values: estimates.append(np.mean(values))
    return np.quantile(estimates,[.025,.975]).tolist() if estimates else [None,None]


def analyze(document,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from arch_dynamics_runner import merge_records
    output=root/'analysis'/document['group']; output.mkdir(parents=True,exist_ok=True)
    runs=[]; summaries=[]
    for run in document['experiments']:
        records=merge_records(root/'runs'/run['experiment_id']/'diagnostics')
        if not records: continue
        event=events(records,run['settings'])
        summaries.append(dict(experiment_id=run['experiment_id'],model=run['model'],rho=run['rho'],snr=run['snr'],
                              train_seed=run['train_seed'],events=event))
        runs.append((run,records,event))
    write_json(output/'events.json',summaries)
    if not runs:
        print('No diagnostic records yet; no figures fabricated.')
        return
    def save(fig,name):
        fig.tight_layout(); fig.savefig(output/(name+'.png'),dpi=140); plt.close(fig)
    grouped=defaultdict(list)
    for run,records,event in runs:
        for membership in run['memberships']:
            grouped[(membership['architecture_family'],run['rho'])].append((run,records,event,membership))
        stem=run['experiment_id']; c=run['settings']; steps=[r['training_step'] for r in records]
        label=' / '.join(m['architecture_label'] for m in run['memberships'])
        title=f'{label}; rho={run["rho"]}; SNR={run["snr"]}; seed={run["train_seed"]}'
        if run['exploratory']: title+='; EXPLORATORY'
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,definition,xkey,ykey in zip(axes,('direct','linear'),
                ('duplicate_fit_ratio','linear_fit_ratio'),('clean_gen_ratio','linear_clean_gen_ratio')):
            x=[mean(r,xkey) for r in records]; y=[mean(r,ykey) for r in records]
            ax.plot(x,y,color='gray',alpha=.6)
            sc=ax.scatter(x,y,c=np.log1p(steps),cmap='viridis')
            ax.add_patch(Rectangle((0,0),c['tau_fit'],c['tau_gen'],alpha=.15,color='green'))
            ax.axvline(c['tau_fit'],ls='--',color='green'); ax.axhline(c['tau_gen'],ls='--',color='green')
            ax.set(xlabel=xkey,ylabel=ykey,title=definition+' BO trajectory')
            ax.set_xscale('symlog',linthresh=.01); ax.set_yscale('symlog',linthresh=.01)
            fig.colorbar(sc,ax=ax,label='log(1 + training step)')
        fig.suptitle(title); save(fig,stem+'_phase_trajectory')
        fig,ax=plt.subplots(figsize=(10,3))
        for i,name in enumerate(('generalize','direct_fit','linear_fit','direct_BO','linear_BO','harmful')):
            ev=event['t_'+name]
            ax.scatter(steps,np.full(len(steps),i),c=['green' if f else 'lightgray' for f in ev['observed_flags']],s=18)
            if ev['step'] is not None: ax.scatter([ev['step']],[i],marker='*',s=100,color='black')
        ax.set_yticks(range(6)); ax.set_yticklabels(('generalize','direct fit','linear fit','direct BO','linear BO','harmful'))
        ax.set(xlabel='Training step (star = stable-window onset)',title=title)
        ax.set_xscale('symlog',linthresh=50); save(fig,stem+'_phase_timeline')
        mech=[r for r in records if r.get('mechanism')]
        if mech:
            for metric in ('layer_response_snr','clean_query_mse_after_layer','attention_entropy','task_vector_probe_r2','clean_target_probe_r2'):
                if 'probe_r2' in metric:
                    grid=np.array([[np.mean([p['layers'][l][metric] for p in r['representation_probes']])
                         for r in mech] for l in range(run['model']['n_layer'])])
                else: grid=np.array([[r['mechanism']['layers'][l][metric]['mean'] for r in mech] for l in range(run['model']['n_layer'])])
                fig,ax=plt.subplots(figsize=(9,4))
                im=ax.imshow(grid,origin='lower',aspect='auto'); fig.colorbar(im,ax=ax,label=metric)
                ax.set_xticks(range(len(mech))); ax.set_xticklabels([r['training_step'] for r in mech],rotation=60)
                ax.set(xlabel='Training step (diagnostic grid)',ylabel='Layer (zero based)',title=title)
                save(fig,stem+'_layer_time_'+metric)
            event_steps={0,records[-1]['training_step']}|{v['step'] for k,v in event.items() if k.startswith('t_') and v['step'] is not None}
            for r in mech:
                if r['training_step'] not in event_steps: continue
                rows=r['mechanism']['heads']
                fig,axes=plt.subplots(1,4,figsize=(16,4))
                for ax,metric in zip(axes,('attention_signal_score','head_noise_response','head_response_snr','attention_entropy')):
                    grid=np.array([h[metric]['mean'] for h in rows]).reshape(run['model']['n_layer'],run['model']['n_head'])
                    im=ax.imshow(grid,origin='lower',aspect='auto'); fig.colorbar(im,ax=ax)
                    ax.set(title=metric,xlabel='Head',ylabel='Layer')
                fig.suptitle(title+f'; step={r["training_step"]}'); save(fig,stem+f'_head_layer_step_{r["training_step"]}')
            rows=mech[-1]['mechanism']['heads']; fig,ax=plt.subplots()
            sc=ax.scatter([h['attention_signal_score']['mean'] for h in rows],
                       [h['head_noise_response']['mean'] for h in rows],c=[h['layer'] for h in rows],
                       s=[10+20*h['head_output_norm']['mean'] for h in rows])
            fig.colorbar(sc,ax=ax,label='Layer'); ax.set(xlabel='Attention signal score (diagnostic)',ylabel='Noise activation-response norm',title=title)
            save(fig,stem+'_head_specialization')
    uncertainty=[]
    for (family,rho),items in grouped.items():
        conditions=sorted(set(item[0]['condition'] for item in items))
        # Six panels: three metric groups by audited benign/harmful columns.
        fig,axes=plt.subplots(3,len(conditions),figsize=(7*len(conditions),11),squeeze=False)
        styles=('solid','dashed','dotted')
        metrics=(('train_loss','clean_gen_ratio','duplicate_fit_ratio'),
                 ('attention_signal_score','attention_noise_score'),('head_response_snr','layer_response_snr'))
        colors={label:f'C{i%10}' for i,label in enumerate(sorted({m['architecture_label'] for _,_,_,m in items}))}
        for col,condition in enumerate(conditions):
            for run,records,ev,m in items:
                if run['condition']!=condition: continue
                for row,keys in enumerate(metrics):
                    for style,key in zip(styles,keys):
                        axes[row,col].plot([r['training_step'] for r in records],[mean(r,key) for r in records],
                            ls=style,color=colors[m['architecture_label']],label=f'{m["architecture_label"]}: {key}; seed={run["train_seed"]}')
                    axes[row,col].set_xscale('symlog',linthresh=50)
                    axes[row,col].set(xlabel='Training step',title=f'{condition}; SNR={run["snr"]}')
                    axes[row,col].legend(fontsize=6)
        fixed={'head_sweep_fixed_width':'width=256, L=12', 'depth_sweep_fixed_width':'width=256, H=8',
               'joint_scale':'Tiny L3/H2/W64; Small L6/H4/W128; Standard L12/H8/W256'}.get(family,'')
        fig.suptitle(f'{family}; {fixed}; rho={rho}; matched. No assumed three stages. Activation responses are diagnostic.')
        save(fig,f'{family}_rho{rho}_figure1_dynamics')
        for metric in ('clean_query_mse','duplicate_fit_ratio','linear_fit_ratio','linear_clean_query_mse',
                       'linear_clean_gen_ratio','heldout_probe_r2','direct_bo_candidate','linear_bo_candidate','head_diversity_index'):
            fig,ax=plt.subplots(figsize=(9,4))
            for run,records,ev,m in items:
                ax.plot([r['training_step'] for r in records],[mean(r,metric) for r in records],
                        label=f'{m["architecture_label"]}; SNR={run["snr"]}; seed={run["train_seed"]}')
            ax.set_xscale('symlog',linthresh=50); ax.set(title=f'{family}, rho={rho}',xlabel='Training step',ylabel=metric); ax.legend(fontsize=6)
            save(fig,f'{family}_rho{rho}_{metric}_dynamics')
        axis='n_head' if family=='head_sweep_fixed_width' else 'n_layer' if family in ('depth_sweep_fixed_width','parameter_matched_depth') else None
        if axis:
            fig,axes=plt.subplots(2,3,figsize=(14,8))
            measures=('t_generalize','t_direct_fit','t_direct_BO','direct_BO_persistence_fraction','final_clean_gen_ratio','head_diversity_index')
            for ax,key in zip(axes.flat,measures):
                for condition in conditions:
                    subset=sorted([v for v in items if v[0]['condition']==condition],key=lambda v:v[0]['model'][axis])
                    ys=[]
                    for run,rs,ev,m in subset:
                        ys.append(ev[key]['step'] if key.startswith('t_') else ev[key] if key in ev else mean(rs[-1],key.replace('final_','')))
                    ax.plot([v[0]['model'][axis] for v in subset],ys,'o-',label=condition)
                ax.set(xlabel=axis,ylabel=key); ax.legend(fontsize=7)
            fig.suptitle(f'{family}; rho={rho}; missing event = not observed'); save(fig,f'{family}_rho{rho}_architecture_summary')
        replicates=defaultdict(list)
        for run,rs,ev,m in items: replicates[(m['architecture_label'],run['snr'])].append(rs)
        for (label,snr),sets in replicates.items():
            common=sorted(set.intersection(*[{r['training_step'] for r in rs} for rs in sets]))
            aligned=[[next(r for r in rs if r['training_step']==s) for s in common] for rs in sets]
            for i,s in enumerate(common):
                uncertainty.append(dict(family=family,rho=rho,snr=snr,label=label,training_step=s,
                    training_runs=len(sets),clean_gen_ratio_ci=hierarchy_ci(aligned,'clean_gen_ratio',i,document['settings']['bootstrap_samples'])))
    write_json(output/'hierarchical_uncertainty.json',uncertainty)
    # Suggestions are reviewable, never used to launch replicates automatically.
    selected=[]; reasons=[]
    for family in ('joint_scale','head_sweep_fixed_width','depth_sweep_fixed_width'):
        candidates=[(r,rs,e) for r,rs,e in runs if any(m['architecture_family']==family for m in r['memberships'])]
        if not candidates: continue
        order=sorted(candidates,key=lambda v:(v[0]['model']['n_head'] if family=='head_sweep_fixed_width' else v[0]['model']['n_layer']))
        picks=[order[0],order[len(order)//2],order[-1]]
        valid=[v for v in candidates if mean(v[1][-1],'clean_gen_ratio') is not None]
        if valid:
            picks += [min(valid,key=lambda v:mean(v[1][-1],'clean_gen_ratio')),max(valid,key=lambda v:mean(v[1][-1],'clean_gen_ratio'))]
        for run,rs,ev in candidates:
            vals=[mean(r,'clean_gen_ratio') for r in rs]
            finite=[v for v in vals if v is not None]
            nonmonotonic=len(finite)>2 and any((b-a)*(cc-b)<0 for a,b,cc in zip(finite,finite[1:],finite[2:]))
            m=rs[-1]['metrics']['clean_gen_ratio']
            wide=m.get('high') is not None and m['high']-m['low']>document['settings']['tau_gen']
            if nonmonotonic or wide: picks.append((run,rs,ev))
        for run,_,_ in picks:
            shape=[run['model'][k] for k in ('n_embd','n_layer','n_head')]
            if shape not in selected: selected.append(shape)
        reasons.append(f'{family}: representative sizes, error extrema, non-monotonic trajectories, wide evaluation uncertainty')
    write_json(output/'architecture_replication_suggestions.json',dict(selected_architectures=selected,reasons=reasons,
        requires_manual_selection=True,train_seeds=[0,1,2,3,4]))
    print('Architecture analysis:',output)
