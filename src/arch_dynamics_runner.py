"""Opt-in architecture dynamics runner; legacy training/evaluation is unchanged."""
import json
import time
from pathlib import Path
import torch
from arch_dynamics_plan import ROOT, load_settings, architecture_table, plan, write_json, digest
from arch_dynamics_diagnostics import prompts, mechanism_batch, merge_mechanism, representation_probes, bootstrap, events


def build(run,device):
    from models import TransformerModel
    torch.manual_seed(run['train_seed'])
    model=TransformerModel(run['settings']['d'],run['settings']['context']+1,**run['model'])
    # New transformers versions otherwise omit attention weights with SDPA.
    if hasattr(model._backbone,'set_attn_implementation'):
        model._backbone.set_attn_implementation('eager')
    return model.to(device)


def merge_records(path):
    if not path.exists(): return []
    # Atomic per-step documents avoid torn append and duplicates after resume.
    return [json.loads(p.read_text()) for p in sorted(path.glob('step_*.json'))]


@torch.no_grad()
def diagnostic(model,run,step,train_loss=None,wall_seconds=0):
    from bo_eval import evaluate_context
    c=run['settings']; device=next(model.parameters()).device
    seed_rows=[]; mechanism=[]; probes=[]
    mode=model.training
    try:
        model.eval()
        for seed in c['eval_seeds']:
            p=prompts(c,run['rho'],run['snr'],seed)
            values={}
            for start in range(0,c['n_eval'],c['eval_batch_size']):
                batch={key:v[start:start+c['eval_batch_size']] for key,v in p.items()}
                b={key:v.to(device) for key,v in batch.items()}
                metrics=evaluate_context(model,b['x'],b['noisy'],b['w'],b['queries'],
                    probe_queries=b['probe'],probe_test_queries=b['probe_test'],
                    run_linear_probe=True,tau_fit=c['tau_fit'],tau_gen=c['tau_gen'],
                    tau_probe_r2=c['tau_probe'],query_batch_size=c['eval_batch_size']*c['n_queries'])
                for key,value in metrics.items():
                    if isinstance(value,torch.Tensor): values.setdefault(key,[]).extend(value.float().cpu().tolist())
                if c['mechanism']: mechanism.append(mechanism_batch(model,batch))
            seed_rows.append(dict(eval_seed=seed,metrics={key:bootstrap(v,c['bootstrap_samples'],seed)
                                                        for key,v in values.items()}))
            if c['mechanism']: probes.append(dict(eval_seed=seed,**representation_probes(model,c,run['rho'],run['snr'],seed)))
    finally:
        model.train(mode)
    means={key:bootstrap([r['metrics'][key]['mean'] for r in seed_rows
                         if r['metrics'][key]['mean'] is not None],c['bootstrap_samples'])
           for key in seed_rows[0]['metrics']}
    # A fixed held-out batch provides train-objective loss at step zero too.
    if train_loss is None:
        p=prompts(c,run['rho'],run['snr'],run['train_seed'],'training_objective',count=c['batch_size'])
        xs=torch.cat([p['x'],p['queries'][:,:1]],1).to(device)
        ys=torch.cat([p['noisy'],(p['queries'][:,:1]*p['w'][:,None]).sum(-1)],1).to(device)
        train_loss=(model(xs,ys)-ys).square().mean().item()
    means['train_loss']=dict(mean=train_loss,low=None,high=None,n=1)
    tokens=step*c['batch_size']*2*(c['context']+1)
    # 6*N*T approximation uses active parameters; unused GPT-2 vocabulary embedding excluded.
    active=sum(p.numel() for n,p in model.named_parameters() if 'wte' not in n)
    return dict(experiment_id=run['experiment_id'],training_step=step,
        examples_seen=step*c['batch_size'],tokens_seen=tokens,
        estimated_training_FLOPs=6*active*tokens,flops_definition='6 * active_parameters * tokens; excludes attention quadratic cost',
        wall_seconds=wall_seconds,metrics=means,evaluation_seeds=seed_rows,
        mechanism=merge_mechanism(mechanism) if mechanism else None,representation_probes=probes,
        uncertainty='bootstrap across evaluation-seed means within one checkpoint; steps are never replicates',
        exploratory=run['exploratory'])


def save_state(path,model,optimizer,scaler,step,run,wall_seconds):
    temp=path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        scaler=scaler.state_dict(),training_step=step,experiment_id=run['experiment_id'],
        wall_seconds=wall_seconds),temp)
    temp.replace(path)


def train_run(run,root,device='cpu',resume=True):
    """Data is stateless by (seed,step), so resume replays identical batches."""
    directory=root/'runs'/run['experiment_id']; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completed.json').exists(): return 'completed (reused)'
    c=run['settings']; model=build(run,device)
    if c['precision']!='float32' and next(model.parameters()).device.type!='cuda':
        raise ValueError('Mixed precision requires CUDA')
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    # ``torch.cuda.amp.GradScaler`` works on the repository's older server
    # environment as well as newer PyTorch releases.  It is inert for bf16.
    scaler=torch.cuda.amp.GradScaler(enabled=c['precision']=='float16')
    step=0; elapsed=0; state=directory/'state.pt'; start=time.monotonic()
    if state.exists():
        if not resume: raise ValueError('Checkpoint exists; use --resume or a different output root')
        s=torch.load(state,map_location=device,weights_only=False)
        if s['experiment_id']!=run['experiment_id']: raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(s['model_state_dict']); optimizer.load_state_dict(s['optimizer_state_dict'])
        scaler.load_state_dict(s['scaler']); step=s['training_step']; elapsed=s['wall_seconds']
    write_json(directory/'run.json',run)
    model.train()
    diagnostics=directory/'diagnostics'
    if step in c['diagnostic_steps'] and not (diagnostics/f'step_{step:09d}.json').exists():
        write_json(diagnostics/f'step_{step:09d}.json',diagnostic(model,run,step,wall_seconds=elapsed))
    try:
        while step<c['train_steps']:
            p=prompts(c,run['rho'],run['snr'],run['train_seed'],'train',step,count=c['batch_size'])
            xs=torch.cat([p['x'],p['queries'][:,:1]],1).to(device)
            # Same causal prefix objective as v1, with a clean independent final query.
            ys=torch.cat([p['noisy'],(p['queries'][:,:1]*p['w'][:,None]).sum(-1)],1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=next(model.parameters()).device.type,
                 dtype=torch.float16 if c['precision']=='float16' else torch.bfloat16,
                 enabled=c['precision']!='float32'):
                loss=(model(xs,ys)-ys).square().mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            step+=1
            wall=elapsed+time.monotonic()-start
            if step in c['diagnostic_steps']:
                record=diagnostic(model,run,step,loss.item(),wall)
                write_json(diagnostics/f'step_{step:09d}.json',record)
                print(run['experiment_id'],step,'loss',loss.item(),flush=True)
            if step%c['save_every']==0 or step==c['train_steps']:
                save_state(state,model,optimizer,scaler,step,run,wall)
        write_json(directory/'completed.json',dict(training_step=step,experiment_id=run['experiment_id']))
    except KeyboardInterrupt:
        save_state(state,model,optimizer,scaler,step,run,elapsed+time.monotonic()-start)
        raise
    return 'completed'


def add_arguments(parser):
    parser.add_argument('--architecture-config',type=Path)
    parser.add_argument('--architecture-audit',type=Path,default=ROOT/'results/bo_matrix/scientific_audit/scientific_audit.json')
    parser.add_argument('--architecture-output',type=Path,default=ROOT/'results/arch_dynamics')
    parser.add_argument('--architecture-selection',type=Path)
    parser.add_argument('--debug-exploratory-snrs',type=float,nargs=2)
    parser.add_argument('--stage-a-inspection',type=Path,help='JSON with inspected=true and matching Stage-A plan signature')


def dispatch(args,action):
    group=args.group[0] if isinstance(args.group,list) else args.group
    if isinstance(args.group,list) and len(args.group)!=1: raise ValueError('Select one architecture group per command')
    c=load_settings(args.architecture_config)
    if getattr(args,'precision',None): c['precision']=args.precision
    root=args.architecture_output
    # Gate before costly parameter-count search, but still emit a readable blocked plan.
    from arch_dynamics_plan import select_conditions
    try:
        select_conditions(args.architecture_audit,c,args.debug_exploratory_snrs)
    except RuntimeError as error:
        write_json(root/'plans'/f'{group}_blocked.json',dict(status='blocked',reason=str(error),settings=c))
        print(error)
        return 2
    cache=root/'architecture_counts.json'
    cache_key=digest(dict(d=c['d'],context=c['context'],parameter_match=group=='arch_parameter_matched_depth',version=1))
    previous=json.loads(cache.read_text()) if cache.exists() else {}
    if previous.get('key')==cache_key: table=previous['rows']
    else:
        table=architecture_table(c,group=='arch_parameter_matched_depth')
        write_json(cache,dict(key=cache_key,rows=table))
    document=plan(group,c,args.architecture_audit,table,args.debug_exploratory_snrs,args.architecture_selection)
    signature=digest(dict(settings=c,conditions=[{k:v for k,v in r.items() if k!='evidence'} for r in document['conditions']]))
    document['stage_a_signature']=signature
    write_json(root/'plans'/f'{group}.json',document)
    if action=='plan' or getattr(args,'dry_run',False):
        print(json.dumps(dict(group=group,runs=document['experiment_count'],stage_a_signature=signature,
            conditions=document['conditions'],output=str(root),status='planned; training not started'),indent=2))
        return 0
    if action=='analyze':
        from arch_dynamics_plot import analyze
        analyze(document,root)
        return 0
    if action=='train' and group not in ('arch_stageA_joint_scale',) and not args.debug_exploratory_snrs:
        approval=json.loads(args.stage_a_inspection.read_text()) if args.stage_a_inspection else {}
        stage_a=root/'plans/arch_stageA_joint_scale.json'
        old=json.loads(stage_a.read_text()) if stage_a.exists() else {}
        if (approval.get('inspected') is not True or approval.get('stage_a_signature')!=signature
            or old.get('stage_a_signature')!=signature
            or not all((root/'runs'/r['experiment_id']/'completed.json').exists() for r in old.get('experiments',[]))
            or not old.get('experiments')):
            raise RuntimeError('Stage A must finish and be inspected; provide --stage-a-inspection')
    if getattr(args,'max_concurrent',1) not in (None,1):
        raise ValueError('Architecture dynamics currently runs sequentially; use one launcher per disjoint plan')
    device=getattr(args,'device','cpu')
    if device=='auto': device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for run in document['experiments']:
        directory=root/'runs'/run['experiment_id']
        if action=='train':
            # Exclusive file creation prevents two writers. Never guess that a lock is stale.
            directory.mkdir(parents=True,exist_ok=True)
            lock=directory/'active.lock'
            try:
                with lock.open('x') as f: f.write(str(__import__('os').getpid()))
            except FileExistsError: raise RuntimeError(f'Run locked: {lock}')
            try: print(train_run(run,root,device,getattr(args,'resume',True)))
            finally: lock.unlink(missing_ok=True)
        elif action=='evaluate':
            state=directory/'state.pt'
            if not state.exists(): raise FileNotFoundError(f'Missing checkpoint: {state}')
            if (directory/'active.lock').exists(): raise RuntimeError('Stop training before final reevaluation')
            model=build(run,device); saved=torch.load(state,map_location=device,weights_only=False)
            if saved['experiment_id']!=run['experiment_id']: raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(saved['model_state_dict'])
            record=diagnostic(model,run,saved['training_step'],wall_seconds=saved['wall_seconds'])
            write_json(directory/'reevaluation'/f'step_{saved["training_step"]:09d}.json',record)
    if action in ('train','evaluate'):
        from arch_dynamics_plot import analyze
        analyze(document,root)
    return 0
