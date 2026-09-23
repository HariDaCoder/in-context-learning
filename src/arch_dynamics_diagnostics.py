"""Matched synthetic prompts and evaluation-only GPT-2 dynamics diagnostics.

Head outputs are captured at c_proj INPUT (concatenated heads, before output
projection/residual addition). Block states are captured before final ln_f.
No attention/activation tensors are serialized.
"""
import hashlib
import math
import torch


def prompt_seed(seed, stream, step=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{stream}:{step}'.encode()).digest()[:8], 'little') % (2**63-1)


def prompts(c, rho, snr, seed, stream='eval', step=0, count=None):
    """CPU generator independent of model initialization, device, and diagnostics."""
    g=torch.Generator().manual_seed(prompt_seed(seed,stream,step))
    b=count or c['n_eval']; k=c['context']; d=c['d']
    x=torch.randn(b,k,d,generator=g)
    for t in range(1,k): x[:,t]=rho*x[:,t-1]+math.sqrt(1-rho*rho)*x[:,t]
    w=torch.randn(b,d,generator=g)/math.sqrt(d)
    clean=(x*w[:,None,:]).sum(-1)
    epsilon=torch.randn(b,k,generator=g)/snr
    q=torch.randn(b,c['n_queries'],d,generator=g)
    return dict(x=x,w=w,epsilon=epsilon,clean=clean,noisy=clean+epsilon,
                xonly=torch.zeros_like(clean),queries=q,
                probe=torch.randn(b,2*d,d,generator=g),
                probe_test=torch.randn(b,max(2*d,32),d,generator=g))


def innovations(x):
    """Sequential orthogonal projection using rank-revealing SVD, in float64."""
    x=x.double(); scores=[]
    for t in range(x.shape[1]):
        v=x[:,t]
        if t:
            _, s, vh=torch.linalg.svd(x[:,:t],full_matrices=False)
            tol=s[:,:1]*max(t,x.shape[-1])*torch.finfo(x.dtype).eps
            coordinates=torch.einsum('bjd,bd->bj',vh,v)*(s>tol)
            v=v-torch.einsum('bj,bjd->bd',coordinates,vh)
        scores.append((v.square().sum(-1)/x[:,t].square().sum(-1).clamp_min(1e-12)).clamp(0,1))
    return torch.stack(scores,1).float()


@torch.no_grad()
def capture(model, x, y, query):
    """Retain only query-position states. Always remove hooks and restore mode."""
    device=next(model.parameters()).device
    x=x.to(device); y=y.to(device); query=query.to(device)
    xs=torch.cat([x,query[:,None]],1)
    ys=torch.cat([y,torch.zeros_like(y[:,:1])],1)
    pos=2*x.shape[1]; heads={}; states={}; handles=[]
    mode=model.training
    try:
        model.eval()
        for l,block in enumerate(model._backbone.h):
            def head_hook(module, inputs, l=l, h=block.attn.num_heads):
                heads[l]=inputs[0][:,pos].detach().reshape(x.shape[0],h,-1).clone()
            def block_hook(module, inputs, output, l=l):
                states[l]=output[0][:,pos].detach().clone()
            handles.append(block.attn.c_proj.register_forward_pre_hook(head_hook))
            handles.append(block.register_forward_hook(block_hook))
        out=model._backbone(inputs_embeds=model._read_in(model._combine(xs,ys)),
                            output_attentions=True,return_dict=True,use_cache=False)
        if out.attentions is None or any(a is None for a in out.attentions):
            raise RuntimeError('Attention diagnostics require GPT-2 eager attention')
        attention=[a[:,:,pos,:pos+1].detach().float() for a in out.attentions]
        pred=model._read_out(out.last_hidden_state[:,pos]).squeeze(-1)
        intermediate=[model._read_out(model._backbone.ln_f(states[l])).squeeze(-1)
                      for l in range(len(states))]
        return dict(heads=heads,states=states,attention=attention,prediction=pred,intermediate=intermediate)
    finally:
        for handle in handles: handle.remove()
        model.train(mode)


def stats(v):
    v=v.detach().double().flatten(); v=v[torch.isfinite(v)]
    if not len(v): return dict(mean=None,variance=None,n=0)
    return dict(mean=v.mean().item(),variance=v.var(unbiased=False).item(),n=len(v))


@torch.no_grad()
def mechanism_batch(model, p):
    q=p['queries'][:,0]; k=p['x'].shape[1]
    captures=[capture(model,p['x'],p[label],q) for label in ('xonly','clean','noisy')]
    zero,clean,noisy=captures
    device=noisy['prediction'].device
    novelty=innovations(p['x']).to(device)
    s=p['clean'].to(device).square(); n=p['epsilon'].to(device).square()
    heads=[]; layers=[]
    for l,a in enumerate(noisy['attention']):
        demo=a[:,:,:2*k].reshape(len(q),a.shape[1],k,2).sum(-1)
        labels=a[:,:,1:2*k:2]
        denom=(labels*(s+n)[:,None,:]).sum(-1).clamp_min(1e-12)
        signal=(labels*s[:,None]).sum(-1)/denom
        noise=(labels*n[:,None]).sum(-1)/denom
        prob=demo/demo.sum(-1,keepdim=True).clamp_min(1e-12)
        entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
        lag=torch.arange(k,0,-1,device=device,dtype=prob.dtype)
        centered=prob-prob.mean(-1,keepdim=True); lag0=lag-lag.mean()
        corr=(centered*lag0).sum(-1)/(centered.square().sum(-1)*lag0.square().sum()).sqrt().clamp_min(1e-12)
        oh=noisy['heads'][l].float()
        sr=(clean['heads'][l]-zero['heads'][l]).float().norm(dim=-1)
        nr=(noisy['heads'][l]-clean['heads'][l]).float().norm(dim=-1)
        cosine=torch.nn.functional.normalize(prob,dim=-1) @ torch.nn.functional.normalize(prob,dim=-1).transpose(1,2)
        left=prob[:,:,None,:]; right=prob[:,None,:,:]; mid=(left+right)/2
        js=.5*((left*(left.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)
               +(right*(right.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1))
        gram=oh @ oh.transpose(1,2)
        rank=gram.diagonal(dim1=-2,dim2=-1).sum(-1).square()/gram.square().sum((-1,-2)).clamp_min(1e-12)
        mask=~torch.eye(a.shape[1],dtype=torch.bool,device=device)
        diversity=1-cosine[:,mask].mean(-1) if a.shape[1]>1 else torch.zeros(len(q),device=device)
        layer_sr=(clean['states'][l]-zero['states'][l]).float().norm(dim=-1)
        layer_nr=(noisy['states'][l]-clean['states'][l]).float().norm(dim=-1)
        target=(q*p['w']).sum(-1).to(device)
        layer=dict(layer=l,head_output_gram=gram.mean(0).cpu().tolist(),
                   pairwise_attention_cosine=cosine.mean(0).cpu().tolist(),
                   pairwise_js_divergence=js.mean(0).cpu().tolist())
        for key,v in dict(head_diversity_index=diversity,head_output_effective_rank=rank,
                          layer_signal_response=layer_sr,layer_noise_response=layer_nr,
                          layer_response_snr=layer_sr/(layer_nr+1e-12),
                          clean_query_mse_after_layer=(noisy['intermediate'][l]-target).square(),
                          attention_entropy=entropy.mean(-1)).items(): layer[key]=stats(v)
        layers.append(layer)
        metrics=dict(attention_signal_score=signal,attention_noise_score=noise,
            x_token_mass=a[:,:,0::2].sum(-1),y_token_mass=labels.sum(-1),
            novelty_weighted_attention=(demo*novelty[:,None]).sum(-1),
            attention_temporal_lag_mean=(prob*lag).sum(-1),attention_lag_correlation=corr,
            recency_mass=demo[:,:,-max(1,k//4):].sum(-1),attention_entropy=entropy,
            effective_attention_context=entropy.exp(),head_signal_response=sr,head_noise_response=nr,
            head_response_snr=sr/(nr+1e-12),head_output_norm=oh.norm(dim=-1))
        for h in range(a.shape[1]):
            heads.append(dict(layer=l,head=h,**{key:stats(v[:,h]) for key,v in metrics.items()}))
    return dict(heads=heads,layers=layers)


def merge_mechanism(batches):
    """Merge scalar moments exactly and small matrices weighted by batch size."""
    result={}
    for kind in ('heads','layers'):
        result[kind]=[]
        for i in range(len(batches[0][kind])):
            out={}
            for key,value in batches[0][kind][i].items():
                values=[b[kind][i][key] for b in batches]
                if isinstance(value,dict):
                    total=sum(v['n'] for v in values)
                    mean=sum(v['mean']*v['n'] for v in values if v['n'])/total if total else None
                    variance=max(0,sum((v['variance']+v['mean']**2)*v['n'] for v in values if v['n'])/total-mean**2) if total else None
                    out[key]=dict(mean=mean,variance=variance,n=total)
                elif isinstance(value,list):
                    weights=[b[kind][i]['attention_entropy']['n'] for b in batches]
                    out[key]=(sum(torch.tensor(v)*w for v,w in zip(values,weights))/sum(weights)).tolist()
                else: out[key]=value
            result[kind].append(out)
    return result


@torch.no_grad()
def representation_probes(model,c,rho,snr,seed):
    datasets=[]
    for stream,count in [('representation_fit',c['probe_train_tasks']),('representation_holdout',c['probe_test_tasks'])]:
        p=prompts(c,rho,snr,seed,stream,count=count)
        states={}
        for start in range(0,count,c['eval_batch_size']):
            z=capture(model,p['x'][start:start+c['eval_batch_size']],p['noisy'][start:start+c['eval_batch_size']],
                      p['queries'][start:start+c['eval_batch_size'],0])
            for l,v in z['states'].items(): states.setdefault(l,[]).append(v.cpu().double())
        datasets.append(({l:torch.cat(v) for l,v in states.items()},
                         torch.cat([(p['queries'][:,0]*p['w']).sum(-1,keepdim=True),p['w']],1).double()))
    (fit,y),(test,yt)=datasets; rows=[]
    for l in fit:
        # Center with training-only means; dual solve avoids width-sized matrices.
        mu=fit[l].mean(0); ym=y.mean(0); x=fit[l]-mu; xt=test[l]-mu
        coef=x.T @ torch.linalg.solve(x@x.T+c['probe_ridge']*torch.eye(len(x)), y-ym)
        pred=xt@coef+ym
        def r2(a,b):
            den=(b-b.mean(0)).square().sum()
            return (1-(a-b).square().sum()/den).item() if den>1e-12 else None
        rows.append(dict(layer=l,clean_target_probe_r2=r2(pred[:,:1],yt[:,:1]),
                         task_vector_probe_r2=r2(pred[:,1:],yt[:,1:])))
    final=rows[-1]['task_vector_probe_r2']
    earliest=next((r['layer'] for r in rows if r['task_vector_probe_r2'] is not None
                  and final is not None and final>0 and r['task_vector_probe_r2']>=c['informative_fraction']*final),None)
    return dict(layers=rows,earliest_informative_layer=earliest,
                fit_stream='representation_fit',holdout_stream='representation_holdout')


def bootstrap(values, repeats=500, seed=0):
    values=torch.as_tensor(values,dtype=torch.float64).flatten()
    values=values[torch.isfinite(values)]
    if not len(values): return dict(mean=None,low=None,high=None,n=0)
    g=torch.Generator().manual_seed(seed)
    samples=values[torch.randint(len(values),(repeats,len(values)),generator=g)].mean(1)
    return dict(mean=values.mean().item(),low=samples.quantile(.025).item(),
                high=samples.quantile(.975).item(),n=len(values))


def events(records,c):
    """Windows count consecutive observed diagnostics, not independent samples."""
    records=sorted(records,key=lambda r:r['training_step']); window=c['stable_window']
    def mean(r,key):
        value=r['metrics'].get(key,{}).get('mean')
        return value if value is not None else float('nan')
    predicates={
        'generalize':lambda r:mean(r,'clean_gen_ratio')<=c['tau_gen'],
        'direct_fit':lambda r:mean(r,'duplicate_fit_ratio')<=c['tau_fit'],
        'linear_fit':lambda r:mean(r,'linear_fit_ratio')<=c['tau_fit'] and mean(r,'heldout_probe_r2')>=c['tau_probe'],
        'direct_BO':lambda r:mean(r,'direct_bo_candidate')>=c['event_frequency'],
        'linear_BO':lambda r:mean(r,'linear_bo_candidate')>=c['event_frequency'],
        'harmful':lambda r:mean(r,'duplicate_fit_ratio')<=c['tau_fit'] and mean(r,'clean_gen_ratio')>c['harmful_threshold'],
    }
    output={}
    for name,predicate in predicates.items():
        flags=[bool(predicate(r)) for r in records]
        onset=next((i for i in range(len(flags)-window+1) if all(flags[i:i+window])),None)
        output['t_'+name]=dict(step=records[onset]['training_step'] if onset is not None else None,
            confirmed_at=records[onset+window-1]['training_step'] if onset is not None else None,
            persistence_fraction=sum(flags[onset:])/len(flags[onset:]) if onset is not None else None,
            persists_to_end=all(flags[onset:]) if onset is not None else None,
            robustness={'stable_window':window,'supported_points':sum(flags)},
            uncertainty='observed diagnostic resolution; not a confidence interval',
            observed_flags=flags)
    g=output['t_generalize']['step']
    for definition in ('direct','linear'):
        f=output['t_'+definition+'_fit']['step']
        output[definition+'_fit_minus_generalize_gap']=f-g if f is not None and g is not None else None
        output[definition+'_BO_persistence_fraction']=output['t_'+definition+'_BO']['persistence_fraction']
    return output
