"""CPU scientific invariants for the opt-in architecture dynamics study."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arch_dynamics_plan import families,plan,load_settings,select_conditions,write_json
from arch_dynamics_diagnostics import prompts,capture,mechanism_batch,representation_probes,events,innovations
from arch_dynamics_runner import build,diagnostic,train_run,merge_records


def settings():
    c=load_settings()
    c.update(d=2,context=3,batch_size=2,train_steps=2,diagnostic_steps=[0,1,2],
        n_eval=2,eval_batch_size=2,n_queries=2,probe_train_tasks=5,probe_test_tasks=3,
        eval_seeds=[1001,1002],bootstrap_samples=10,save_every=1,mechanism=True)
    return c


def run_spec():
    return dict(experiment_id='unit_cpu',settings=settings(),model=dict(n_embd=8,n_layer=2,n_head=2),
                rho=.9,snr=2.,train_seed=0,exploratory=True,condition='exploratory',
                memberships=[dict(architecture_family='joint_scale',architecture_label='Unit')])


class ArchitectureDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_joint_scale_head_dimension(self):
        self.assertEqual([s['head_dim'] for s in families()['joint_scale']],[32]*3)

    def test_fixed_head_sweep(self):
        for r in families()['head_sweep_fixed_width']:
            self.assertEqual((r['n_embd'],r['n_layer']),(256,12))
            self.assertEqual(r['n_embd']%r['n_head'],0)

    def test_depth_sweep(self):
        rows=families()['depth_sweep_fixed_width']
        self.assertEqual([r['n_layer'] for r in rows],[1,3,6,12,24])
        self.assertTrue(all((r['n_embd'],r['n_head'])==(256,8) for r in rows))

    def test_exact_head_parameter_counts(self):
        from bo_architecture import ArchitectureSpec,instantiate_and_count
        # Meta allocation still instantiates every module without allocating large tensors.
        from models import build_model
        def factory(config):
            with torch.device('meta'): return build_model(config)
        counts=[instantiate_and_count(ArchitectureSpec(256,12,h),20,41,model_factory=factory)
                for h in (1,2,4,8,16)]
        self.assertEqual(len(set(counts)),1)

    def test_common_random_numbers_independent_of_initialization(self):
        c=settings(); p=prompts(c,.9,2,0,'train',1)
        build(run_spec(),'cpu')
        q=prompts(c,.9,2,0,'train',1)
        for key in p: self.assertTrue(torch.equal(p[key],q[key]))

    def test_paired_prompts(self):
        p=prompts(settings(),.9,2,1001)
        self.assertTrue(torch.equal(p['noisy'],p['clean']+p['epsilon']))
        self.assertEqual(p['xonly'].count_nonzero(),0)

    def test_capture_prediction_invariance_and_cleanup(self):
        run=run_spec(); model=build(run,'cpu'); model.train()
        p=prompts(settings(),.9,2,1001)
        xs=torch.cat([p['x'],p['queries'][:,:1]],1); ys=torch.cat([p['noisy'],torch.zeros(2,1)],1)
        expected=model(xs,ys)[:,-1].detach()
        z=capture(model,p['x'],p['noisy'],p['queries'][:,0])
        self.assertTrue(torch.allclose(expected,z['prediction'],atol=1e-6))
        self.assertTrue(torch.allclose(expected,z['intermediate'][-1],atol=1e-6))
        self.assertTrue(model.training)
        self.assertFalse(z['prediction'].requires_grad)
        self.assertTrue(all(not b._forward_hooks and not b.attn.c_proj._forward_pre_hooks for b in model._backbone.h))

    def test_degenerate_activation_decomposition(self):
        model=build(run_spec(),'cpu'); p=prompts(settings(),0,2,1001)
        p['epsilon'].zero_(); p['noisy']=p['clean'].clone()
        result=mechanism_batch(model,p)
        self.assertTrue(all(h['head_noise_response']['mean']==0 for h in result['heads']))
        p['clean'].zero_(); p['noisy'].zero_()
        result=mechanism_batch(model,p)
        self.assertTrue(all(h['head_signal_response']['mean']==0 for h in result['heads']))

    def test_diagnostics_do_not_change_parameters_or_rng(self):
        run=run_spec(); model=build(run,'cpu'); model.train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        record=diagnostic(model,run,0)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIn('linear_bo_candidate',record['metrics'])

    def test_probe_sets_disjoint(self):
        p=prompts(settings(),0,2,1001,'representation_fit')
        q=prompts(settings(),0,2,1001,'representation_holdout')
        self.assertFalse(torch.equal(p['x'],q['x']))
        self.assertFalse(torch.equal(p['w'],q['w']))

    def test_stable_events(self):
        c=settings(); c['stable_window']=2
        def row(step,value): return dict(training_step=step,metrics={'clean_gen_ratio':{'mean':value}})
        e=events([row(0,1),row(50,0),row(100,1)],c)
        self.assertIsNone(e['t_generalize']['step'])
        e=events([row(0,1),row(50,0),row(100,0),row(200,1)],c)
        self.assertEqual(e['t_generalize']['step'],50)
        self.assertEqual(e['t_generalize']['confirmed_at'],100)
        self.assertFalse(e['t_generalize']['persists_to_end'])

    def test_gate_requires_matched_verified_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.json'; write_json(p,{'recommendation':'PROCEED_MAIN_BO','rows':[]})
            with self.assertRaisesRegex(RuntimeError,'ARCHITECTURE_BO_BLOCKED'): select_conditions(p,settings())
            rows=[]
            for rho in (0.,.9):
                for snr,gen in [(1.,2.),(2.,.01)]:
                    rows.append(dict(rho_x=rho,rho_e=0,d=2,k=3,snr=snr,fully_matched=True,
                        duplicate_fit_ratio=.01,clean_gen_ratio=gen,direct_bo_frequency=1 if gen<1 else 0))
            write_json(p,{'rows':rows}); self.assertEqual(len(select_conditions(p,settings())),4)
            for row in rows: row['fully_matched']=False
            write_json(p,{'rows':rows})
            with self.assertRaises(RuntimeError): select_conditions(p,settings())

    def test_outside_context_and_dimension_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.json'
            write_json(p,{'rows':[dict(rho_x=0,rho_e=0,d=2,k=80,snr=2,fully_matched=True,direct_bo_frequency=1)]})
            with self.assertRaises(RuntimeError): select_conditions(p,settings())

    def test_dedup_standard_and_counts(self):
        table=[dict(r,architecture_family=f,exact_parameter_count=1) for f,rs in families().items() for r in rs]
        all_ids=set()
        for group,count in [('arch_stageA_joint_scale',12),('arch_stageB_head_sweep',20),('arch_stageC_depth_sweep',20)]:
            result=plan(group,settings(),'missing',table,[1.,2.])
            self.assertEqual(result['experiment_count'],count)
            all_ids.update(r['experiment_id'] for r in result['experiments'])
        self.assertEqual(len(all_ids),44)

    def test_innovation_rank_sanity(self):
        x=torch.tensor([[[1.,0.],[2.,0.],[0.,1.],[1.,1.]]])
        self.assertTrue(torch.allclose(innovations(x),torch.tensor([[1.,0.,1.,0.]]),atol=1e-6))

    def test_cpu_train_save_reuse_and_plots(self):
        from arch_dynamics_plot import analyze
        run=run_spec()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(train_run(run,root),'completed')
            self.assertEqual(train_run(run,root),'completed (reused)')
            rs=merge_records(root/'runs/unit_cpu/diagnostics')
            self.assertEqual([r['training_step'] for r in rs],[0,1,2])
            analyze(dict(group='unit',settings=run['settings'],experiments=[run]),root)
            self.assertTrue((root/'analysis/unit/unit_cpu_phase_trajectory.png').exists())
            self.assertTrue((root/'analysis/unit/events.json').exists())


if __name__=='__main__': unittest.main()
