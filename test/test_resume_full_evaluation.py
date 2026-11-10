"""Exercise frozen evaluation preparation without importing the ML runtime."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('resume_full_evaluation', ROOT/'scripts/resume_full_evaluation.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode())


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT/'.cache/resume_full_tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=cache)
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(cache.resolve()))
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)/'project with spaces'
        self.project.mkdir()
        self.output = self.project/'result/original_run'
        self.output.mkdir(parents=True)
        self.normalizer = b'{"scales": [1, 2, 3]}\n'
        norm = self.project/'configs/normalization.json'
        norm.parent.mkdir()
        norm.write_bytes(self.normalizer)
        full = {
            'experiment_name':'fixture', 'seed':11, 'device':'cuda',
            'paths':{'instances_root':'data/instances/v3', 'manifests_root':'data/manifests/v3'},
            'dataset':{}, 'generator':{}, 'environment':{}, 'network':{}, 'ppo':{},
            'reward':{}, 'preference':{}, 'evaluation':{}, 'runtime_manifest':{},
            'workload_objective':{'mode':'geometric_owa','q':.5},
            'training':{'episodes':2000,'episodes_per_update':20,'validation_interval_epochs':5},
            'objective_scalarizer':{'normalization_manifest':'configs/normalization.json',
                                    'normalization_manifest_sha256':sha(self.normalizer)},
        }
        instance, certificate = b'{"case":"C01"}\n', b'{"valid":true}\n'
        self.instance = self.project/'data/instances/v3/test/C01.json'
        self.instance.parent.mkdir(parents=True)
        self.instance.write_bytes(instance)
        (self.instance.parent/'C01.certificate.json').write_bytes(certificate)
        manifest = {'split':'test','files':[{'path':'test/C01.json','sha256':sha(instance),
            'certificate':'test/C01.certificate.json','certificate_sha256':sha(certificate)}]}
        self.manifest_path = self.project/'data/manifests/v3/test/manifest.json'
        save_json(self.manifest_path, manifest)
        self.sources = {'scripts/experiment_suite.py':b'# frozen evaluator\r\n',
                        'configs/config.py':b'# frozen config loader\n'}
        self.payload = {
            'suite':{'seeds':[11,23,37,53,71], 'base_config':'configs/comparators/suite.json',
                     'arms':{'full':{'method':'ppo','network':{}}}},
            'sources':{n:sha(b) for n,b in self.sources.items()},
            'configs':{'full':full}, 'data':{'test':manifest},
        }
        self.write_snapshot()
        for seed in self.payload['suite']['seeds']:
            directory = self.output/'training/full'/str(seed)
            save_json(directory/'summary.json', {'budget_completed':True,'episodes_started':2000,
                'epochs':100,'validation_count':20,'method':'ppo','seed':seed})
            (directory/'best_checkpoint.pt').write_bytes(b'fixture checkpoint; never executed')

    def write_snapshot(self, *, extras=None):
        with zipfile.ZipFile(self.output/'source.zip','w') as archive:
            for name, raw in self.sources.items():
                archive.writestr(name,raw)
            archive.writestr('experiment_identity.json',json.dumps(self.payload))
            for name, raw in (extras or {}).items():
                archive.writestr(name,raw)
        save_json(self.output/'preflight.json',{'status':'passed','identity':fingerprint(self.payload),
            'source_hashes':self.payload['sources'],'configs':self.payload['configs'],
            'suite':self.payload['suite'],'data_manifests':self.payload['data']})

    def add_saved_arms(self, arms):
        networks = {'no_graph': {'encoder_variant':'node_mlp_pool'},
                    'no_direct': {'direct_features_enabled':False},
                    'no_residual': {'residual_enabled':False},
                    'rule_ddqn': {}, 'dcan': {}}
        for arm in arms:
            method = arm if arm in ('rule_ddqn', 'dcan') else 'ppo'
            self.payload['suite']['arms'][arm] = {'method':method, 'network':networks[arm]}
            config = json.loads(json.dumps(self.payload['configs']['full']))
            config['network'].update(networks[arm])
            self.payload['configs'][arm] = config
            for seed in self.payload['suite']['seeds']:
                directory = self.output/'training'/arm/str(seed)
                save_json(directory/'summary.json', {'budget_completed':True,'episodes_started':2000,
                    'epochs':100,'validation_count':20,'method':method,'seed':seed})
                (directory/'best_checkpoint.pt').write_bytes(b'fixture checkpoint; never executed')
        self.write_snapshot()

    def remove_full_training(self):
        directory = self.output/'training/full'
        for seed in self.payload['suite']['seeds']:
            (directory/str(seed)/'summary.json').unlink()
            (directory/str(seed)/'best_checkpoint.pt').unlink()
            (directory/str(seed)).rmdir()
        directory.rmdir()

    def test_restores_missing_json_presets_and_exact_source_bytes(self):
        originals = {p.relative_to(self.project):sha(p.read_bytes()) for p in self.project.rglob('*') if p.is_file()}
        frozen, identity, seeds = helper.prepare(self.project,self.output)
        self.assertTrue(frozen.is_relative_to(self.project/'.cache'))
        self.assertEqual(identity,fingerprint(self.payload))
        self.assertEqual(seeds,[11,23,37,53,71])
        for name,raw in self.sources.items():
            self.assertEqual((frozen/name).read_bytes(),raw)
        for name in ['configs/default.json','configs/comparators/suite.json']:
            self.assertEqual(json.loads((frozen/name).read_text()),self.payload['configs']['full'])
        self.assertEqual((frozen/'configs/normalization.json').read_bytes(),self.normalizer)
        self.assertEqual((frozen/'data/instances/v3/test/C01.json').read_bytes(),self.instance.read_bytes())
        for name,expected in originals.items():
            self.assertEqual(sha((self.project/name).read_bytes()),expected)

    def test_rejects_corrupted_snapshot_identity(self):
        pre = json.loads((self.output/'preflight.json').read_text())
        pre['identity']='0'*64
        save_json(self.output/'preflight.json',pre)
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_changed_archived_source(self):
        self.sources['scripts/experiment_suite.py']=b'# different source\n'
        self.write_snapshot()
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_incomplete_training(self):
        path=self.output/'training/full/37/summary.json'
        summary=json.loads(path.read_text());summary['epochs']=99
        save_json(path,summary)
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_changed_fixed_instance(self):
        self.instance.write_bytes(b'{"case":"other"}\n')
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_changed_normalization(self):
        (self.project/'configs/normalization.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_changed_dataset_manifest(self):
        manifest=json.loads(self.manifest_path.read_text());manifest['split']='other'
        save_json(self.manifest_path,manifest)
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)

    def test_rejects_path_traversal(self):
        for name in ['../outside','..\\outside','C:/outside','/outside','folder/../../outside']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                helper.relative_file(self.project,name)

    def test_rejects_archive_path_traversal(self):
        self.write_snapshot(extras={'../escaped.py':b'# escaped'})
        with self.assertRaises(ValueError):
            helper.prepare(self.project,self.output)
        self.assertFalse((self.project/'.cache/escaped.py').exists())

    def invoke_main(self, *args):
        with patch.object(helper.sys,'argv',['resume_full_evaluation.py','--project-root',str(self.project),*args]):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                helper.main()

    def test_prepare_only_does_not_start_subprocesses(self):
        with patch.object(helper.subprocess,'run') as run:
            self.invoke_main('--prepare-only')
        run.assert_not_called()

    def test_ambiguous_model_roots_require_selection(self):
        extra=self.project/'result/second_run/training/full/11/best_checkpoint.pt'
        extra.parent.mkdir(parents=True);extra.write_bytes(b'second model')
        with self.assertRaises(SystemExit),patch.object(helper.subprocess,'run') as run:
            self.invoke_main('--prepare-only')
        run.assert_not_called()

    def test_verification_failure_stops_before_evaluation(self):
        with patch.object(helper.subprocess,'run',side_effect=helper.subprocess.CalledProcessError(3,['identity'])) as run:
            with self.assertRaises(helper.subprocess.CalledProcessError):
                self.invoke_main()
        self.assertEqual(run.call_count,1)

    def test_verifies_identity_then_evaluates_full_with_saved_seeds(self):
        with patch.object(helper.subprocess,'run') as run:
            self.invoke_main()
        self.assertEqual(run.call_count,2)
        verification,evaluation=run.call_args_list
        self.assertIn('-c',verification.args[0])
        command=evaluation.args[0]
        self.assertEqual(command[command.index('--stage')+1],'evaluate')
        self.assertEqual(command[command.index('--arms')+1],'full')
        self.assertEqual(command[command.index('--task')+1],'both')
        start=command.index('--seeds')+1
        self.assertEqual(command[start:start+5],['11','23','37','53','71'])
        self.assertEqual(Path(command[command.index('--output-dir')+1]),self.output)
        self.assertEqual(verification.kwargs['cwd'],evaluation.kwargs['cwd'])

    def test_discovers_and_prepares_three_ablations_without_full_training(self):
        arms = ['no_graph','no_direct','no_residual']
        self.add_saved_arms(arms)
        self.remove_full_training()
        self.assertEqual(helper._discover_output(self.project, arms), self.output)
        frozen, identity, seeds = helper.prepare(self.project, self.output, arms)
        self.assertEqual(identity, fingerprint(self.payload))
        self.assertEqual(seeds, self.payload['suite']['seeds'])
        self.assertEqual(json.loads((frozen/'configs/default.json').read_text()), self.payload['configs']['full'])
        with patch.object(helper.subprocess,'run') as run:
            self.invoke_main('--arms', *arms)
        verification, evaluation = run.call_args_list
        command = evaluation.args[0]
        start, end = command.index('--arms') + 1, command.index('--seeds')
        self.assertEqual(command[start:end], arms)
        self.assertEqual(json.loads(verification.args[0][-1]), arms)
        self.assertEqual(command[command.index('--task')+1], 'both')
        self.assertEqual(Path(command[command.index('--output-dir')+1]), self.output)

    def test_selected_arm_validation_precedes_subprocesses(self):
        arms = ['no_graph','no_direct','no_residual']
        self.add_saved_arms(arms)
        path = self.output/'training/no_direct/53/summary.json'
        summary = json.loads(path.read_text()); summary['budget_completed'] = False
        save_json(path, summary)
        with patch.object(helper.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                self.invoke_main('--output-dir', str(self.output), '--arms', *arms)
        run.assert_not_called()

    def test_rejects_unknown_duplicate_and_unsaved_selected_arms(self):
        for arms in (['unknown'], ['full','full'], ['no_graph']):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                helper.prepare(self.project, self.output, arms)

    def test_selected_methods_are_validated_in_training_summaries(self):
        self.add_saved_arms(['rule_ddqn','dcan'])
        helper.prepare(self.project, self.output, ['rule_ddqn','dcan'])
        path = self.output/'training/dcan/71/summary.json'
        summary = json.loads(path.read_text()); summary['method'] = 'ppo'
        save_json(path, summary)
        with self.assertRaises(ValueError):
            helper.prepare(self.project, self.output, ['dcan'])

    def test_selected_arm_definitions_and_saved_configs_are_required(self):
        self.add_saved_arms(['no_graph'])
        del self.payload['configs']['no_graph']
        self.write_snapshot()
        with self.assertRaises(ValueError):
            helper.prepare(self.project, self.output, ['no_graph'])
        self.payload['configs']['no_graph'] = self.payload['configs']['full']
        self.payload['suite']['arms']['no_graph']['method'] = 'rule_ddqn'
        self.write_snapshot()
        with self.assertRaises(ValueError):
            helper.prepare(self.project, self.output, ['no_graph'])

    def runtime_stub(self, identity):
        stub=self.project/'runtime_stub'
        for package in ['configs','scripts']:
            (stub/package).mkdir(parents=True)
            (stub/package/'__init__.py').write_text('',encoding='utf-8')
        (stub/'configs/experiment_suite.py').write_text(
            "def load_suite(): return {'seeds': [11, 23, 37, 53, 71], 'execution': {'torch_threads': 1}, "
            "'arms': {'full': {'method': 'ppo'}, 'no_graph': {'method': 'ppo'}, "
            "'no_direct': {'method': 'ppo'}, 'no_residual': {'method': 'ppo'}}}\n"
            f"def suite_identity(suite, device): return {identity!r}\n",encoding='utf-8')
        (stub/'torch.py').write_text(
            "threads = 7\n"
            "def set_num_threads(value):\n    global threads\n    threads = value\n"
            "def get_num_threads(): return threads\n",encoding='utf-8')
        (stub/'scripts/experiment_suite.py').write_text(
            "def check_preflight(root, identity): return None\n"
            "def hardware():\n    import torch\n    return {'cpu': 'fixture CPU', 'torch_threads': torch.get_num_threads()}\n",encoding='utf-8')
        return stub

    def test_identity_guard_survives_optimized_python(self):
        stub=self.runtime_stub('wrong-identity')
        result=subprocess.run([helper.sys.executable,'-O','-c',helper._VERIFY_RUNTIME,
            'expected-identity',str(self.output),'cuda','[11,23,37,53,71]'],
            cwd=stub,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Frozen effective identity differs',result.stderr)
        self.assertFalse((self.output/'evaluation_runtime').exists())

    def test_verified_runtime_records_hardware_and_selected_requests(self):
        stub=self.runtime_stub('expected-identity')
        result=subprocess.run([helper.sys.executable,'-O','-c',helper._VERIFY_RUNTIME,
            'expected-identity',str(self.output),'cuda','[11,23,37,53,71]'],
            cwd=stub,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        files=list((self.output/'evaluation_runtime').glob('full_*.json'))
        self.assertEqual(len(files),1)
        metadata=json.loads(files[0].read_text())
        self.assertEqual(metadata['source_identity'],'expected-identity')
        self.assertEqual(metadata['hardware'],{'cpu':'fixture CPU','torch_threads':1})
        self.assertEqual(metadata['arms'],['full'])
        self.assertEqual(metadata['seeds'],[11,23,37,53,71])
        self.assertEqual(metadata['stage'],'evaluate')

    def test_runtime_records_three_selected_ablations(self):
        stub = self.runtime_stub('expected-identity')
        arms = ['no_graph','no_direct','no_residual']
        result = subprocess.run([helper.sys.executable,'-O','-c',helper._VERIFY_RUNTIME,
            'expected-identity',str(self.output),'cuda','[11,23,37,53,71]',json.dumps(arms)],
            cwd=stub,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        files = list((self.output/'evaluation_runtime').glob('*.json'))
        self.assertEqual(len(files),1)
        metadata = json.loads(files[0].read_text())
        self.assertEqual(metadata['arms'],arms)
        self.assertEqual(metadata['task'],'both')
        self.assertEqual(metadata['hardware']['torch_threads'],1)


if __name__=='__main__':
    unittest.main()
