"""Contracts that matter for independent post-training challenge evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import random

import pytest

from configs import load_config, project_path
from data.challenge import (ChallengeDataset, NAMES, _stratum_task, apply_scale, atomic_json,
                            audit_suites, build_candidate, build_suites, diagnose, digest,
                            load_spec, load_suites, read_sealed, replay_certificate, seal)
from data.dataset import canonical_json_bytes, sha256_bytes
from data.generate_orders import InstanceGenerator, _quantize
from data.models import instance_to_dict, load_instance_yaml


@pytest.fixture(scope="module")
def config():
    return load_config("configs/default.json")


@pytest.fixture(scope="module")
def witness(config):
    instance, latent, precheck, reasons = build_candidate(config, NAMES[0], "worker_bottleneck", "balanced_cost", 6008000)
    assert not reasons and precheck["passed"]
    result = diagnose(instance, config)
    assert result["certificate"]
    return instance, result


def test_old_generator_bytes_unchanged(config):
    expected = {
        "easy": "0f4923237335a1078ea0c5feccaf4e1b1e69d20d2908643f8ab990dffe1fe788",
        "balanced": "4f4f5e85647caa212c326fae0f292b33ae2ddd3b181fae50dd5dcf9ed8721378",
        "machine_bottleneck": "a7fe6b1fd7c4afee4c53f7c58847f1fd2005c8a31ebb687fc62b712a086cd264",
        "reconfiguration_bottleneck": "314f0f7768c9bc59283b506f1f79eadc3fd4a70527c8144fbece62e4b96e7953",
        "worker_bottleneck": "779298f52c2ee307cbede5c6490ea6e735edc8ba6be8857d73ac699c4878bff9",
        "fatigue_bottleneck": "eeb88d892572a019ebc7173cf607642a446a883180b82497af7379512d0bd6ec",
        "high_arrival_pressure": "480eb16ce0f982573df90f23284998a234bb6c8174dd23848062be0138d44fff"}
    generator = InstanceGenerator(load_instance_yaml(project_path(config["paths"]["fixed_instance"])), config["generator"], config=config)
    for pressure, expected_hash in expected.items():
        instance = generator._build_candidate(rng=random.Random(123456), seed=3000000, split="test",
                  pressure_type=pressure, cost_profile="balanced_cost", ood_factor=None, attempt=0)
        assert sha256_bytes(canonical_json_bytes(instance_to_dict(instance))) == expected_hash


@pytest.mark.parametrize("defect", ["illegal", "incomplete", "unsafe", "horizon", "physical", "schedule"])
def test_certificate_rejects_invalid_evidence(config, witness, defect):
    instance, result = witness
    certificate = deepcopy(result["certificate"])
    if defect == "illegal":
        certificate["actions"][0] = -1
    elif defect == "incomplete":
        certificate["actions"] = certificate["actions"][:-1]
    elif defect == "unsafe":
        instance = replace(instance, fatigue=replace(instance.fatigue, maximum_safe_fatigue=.01))
        from data.challenge import physical_hash
        certificate["physical_sha256"] = physical_hash(instance)
    elif defect == "horizon":
        instance = replace(instance, horizon=result["heuristic_metrics"]["heuristic_makespan"] - instance.resolution)
        from data.challenge import physical_hash
        certificate["physical_sha256"] = physical_hash(instance)
    elif defect == "physical":
        certificate["physical_sha256"] = "wrong"
    else:
        certificate["schedule_sha256"] = "wrong"
    certificate["action_sha256"] = digest(certificate["actions"])
    with pytest.raises(ValueError):
        replay_certificate(instance, config, certificate)


def test_scale_pairs_share_all_other_physics_and_unrounded_latent(config):
    base, latent, _, reasons = build_candidate(config, NAMES[1], "reconfiguration_bottleneck", "balanced_cost", 7000000)
    assert not reasons
    stripped = []
    for scale in (1.4,1.6,1.8,2.):
        instance = apply_scale(base, latent, scale)
        times = {(t["machine"],t["module"],t["phase"]):t["unscaled"] for t in latent["unscaled_times"]}
        for machine in instance.machines:
            for module,p in machine.module_parameters.items():
                assert p.installation_base_time == _quantize(times[machine.id,module,"installation"]*scale,instance.resolution)
                assert p.disassembly_base_time == _quantize(times[machine.id,module,"disassembly"]*scale,instance.resolution)
        raw = instance_to_dict(instance)
        raw.pop("instance_id")
        for m in raw["machines"]:
            for p in m["module_parameters"].values():
                p.pop("installation_base_time");p.pop("disassembly_base_time")
        stripped.append(raw)
    assert all(r==stripped[0] for r in stripped)


def test_unknown_high_scales_retained_and_hard_quota_not_relaxed(config, tmp_path, monkeypatch):
    spec = load_spec()
    spec.update(instances_root=str(tmp_path), max_candidates_per_stratum=1)
    def unknown(instance, config):
        return {"feasibility_status":"unknown", "certificate":None,
                "diagnostic_terminal_reason":"horizon", "heuristic_metrics":{},"diagnostic_actions":[]}
    monkeypatch.setattr("data.challenge.diagnose",unknown)
    # Even a failed necessary-condition check at a high scale is retained.
    import data.challenge as module
    original = module.cheap_feasibility_precheck
    class Failed:
        def to_dict(self):return {"passed":False,"reason_codes":["high_scale_bound"]}
    monkeypatch.setattr(module,"cheap_feasibility_precheck",lambda i,*a: original(i,*a) if not i.instance_id.endswith("scale2.0") else Failed())
    out = _stratum_task((spec,config,"test",NAMES[1],"reconfiguration_bottleneck","balanced_cost",0,1))
    assert out["complete"] and len(out["accepted"][0]["members"])==4
    assert not out["accepted"][0]["members"][-1]["metadata"]["feasibility_precheck"]["passed"]
    hard = _stratum_task((spec,config,"test",NAMES[0],"worker_bottleneck","balanced_cost",0,1))
    assert not hard["complete"] and not hard["accepted"]
    with pytest.raises(ValueError,match="identity"):
        _stratum_task((spec,config,"different",NAMES[0],"worker_bottleneck","balanced_cost",0,1))


@pytest.fixture(scope="module")
def small_suites(tmp_path_factory):
    spec=load_spec();root=tmp_path_factory.mktemp("challenge")
    spec.update(instances_root=str(root/"instances"),manifests_root=str(root/"manifests"),
                pressures=["worker_bottleneck"],cost_quotas={"balanced_cost":1},max_candidates_per_stratum=3)
    build_suites(spec,workers=1)
    return spec,root


def test_resume_and_worker_count_do_not_change_frozen_files(small_suites):
    spec,root=small_suites
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*.json")}
    build_suites(spec,workers=2)
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*.json")}


def test_fresh_parallel_generation_selects_identical_members(small_suites,tmp_path):
    spec,root=small_suites
    parallel=deepcopy(spec)
    parallel.update(instances_root=str(tmp_path/"instances"),manifests_root=str(tmp_path/"manifests"))
    build_suites(parallel,workers=2)
    a=load_suites(root/"manifests"/"suite.json")
    b=load_suites(tmp_path/"manifests"/"suite.json")
    for left,right in zip(a,b):
        assert [r.instance.instance_id for r in left]==[r.instance.instance_id for r in right]
        assert [r.to_dict() for r in left]==[r.to_dict() for r in right]


@pytest.mark.parametrize("target",["manifest","instance","certificate","ledger"])
def test_tampering_rejected(small_suites,config,target):
    spec,root=small_suites
    mp=root/"manifests"/NAMES[0]/"manifest.json"
    dataset=ChallengeDataset(mp)
    if target=="manifest":path=mp
    elif target=="instance":path=dataset.instances_root/dataset.manifest["files"][0]["path"]
    elif target=="certificate":path=dataset.instances_root/dataset[0].metadata["certificate_file"]
    else:path=dataset.instances_root/"candidates"/dataset.manifest["candidate_receipts"][0]["path"]
    before=path.read_bytes()
    try:
        raw=json.loads(before);raw["tampered"]=True;path.write_text(json.dumps(raw),encoding="utf-8")
        with pytest.raises(ValueError):
            ds=load_suites(root/"manifests"/"suite.json")
            audit_suites(ds,config,replay=False)
    finally:path.write_bytes(before)


def _sample(i,q,success=True):
    return {"instance_id":"i","preference_key":"p","sampling_repeat":i,
            "metrics":{"task_succeeded":success,"truncated":False,"preference_quality_score":q,"schedule_violations":[]}}


def test_native_best_three_and_cluster_units():
    from analysis.challenge_study import delivery_rows,bootstrap_interval
    samples=[_sample(0,.1,False),_sample(1,.4),_sample(2,.4)]
    assert delivery_rows(samples)==1
    rows=[{"suite":NAMES[1],"base_instance_id":b,"instance_id":str(i),"quality":v} for b,v in [("a",.2),("b",.6)] for i in range(12)]
    summary=bootstrap_interval(rows,"quality",repeats=1000)
    assert summary["independent_units"]==2 and summary["mean"]==pytest.approx(.4)


def test_incomplete_request_is_not_reusable():
    from analysis.challenge_study import validate_request
    value={"request_sha256":"a","raw":[],"selected_repeat":0}
    assert not validate_request(value,"a",policy="ppo",preference_key="p",iid="i")
    with pytest.raises(ValueError,match="identity"):
        validate_request(value,"b",policy="ppo",preference_key="p",iid="i")


def test_parallel_capture_preserves_native_sampled_actions(config,witness):
    import torch
    from agent.ppo import PPOAgent,build_actor_critic
    from agent.ppo.parallel import ParallelEpisodeRunner
    from data.dataset import GeneratedInstanceRecord
    from environment import AssemblySchedulingEnv,PreferenceContext
    from eval import EvaluationPolicy,evaluate_instance
    from utils import action_trace_sha256
    c=deepcopy(config);c['device']='cpu';c['environment']['max_decisions']=8
    torch.set_num_threads(1)
    instance=witness[0];obs=AssemblySchedulingEnv(c).reset(instance)
    agent=PPOAgent(build_actor_critic(obs,c['network']),c['ppo'],device='cpu')
    pref=PreferenceContext.from_input((1,0,0))
    policy=EvaluationPolicy(c,policy_name='ppo',bootstrap_observation=obs,ppo_agent=agent,decode_mode='sampled',sampling_seed=100011)
    env,metrics=evaluate_instance(c,instance=instance,policy_name='ppo',prepared_policy=policy,preference=pref,decode_mode='sampled')
    with ParallelEpisodeRunner(config=c,template=instance,episode_count=1,worker_count=1) as runner:
        result=runner.evaluate_records(agent,[GeneratedInstanceRecord(instance,{})],preferences=[pref],sampling_seed=100011,deterministic=False,capture_trajectories=True)[0]
        ordinary=runner.evaluate_records(agent,[GeneratedInstanceRecord(instance,{})],preferences=[pref],sampling_seed=100011,deterministic=False)[0]
    assert list(result.actions)==policy._episode_actions
    assert result.action_trace_sha256==action_trace_sha256(list(result.actions))
    assert result.schedule==env.schedule_log and result.reconfigurations==env.reconfiguration_log
    assert result.metrics==ordinary.metrics
    assert ordinary.actions==() and ordinary.schedule==[] and ordinary.reconfigurations==[]


def test_heuristic_rescore_accepts_context_and_preserves_reference(config,witness):
    from analysis.challenge_study import rescore_heuristic_sample
    from environment import PreferenceContext
    from environment.types import metrics_flow_objective,terminal_quality_score
    from environment.workload import metrics_workload_objective
    instance,result=witness
    metrics=replay_certificate(instance,config,result['certificate'])
    source={'preference':[1.,0.,0.],'metrics':metrics,'success':True}
    before=deepcopy(source)
    preference=PreferenceContext.from_input((0,1,0))
    sample=rescore_heuristic_sample(source,preference,config)
    assert source==before
    assert sample['source_trajectory_preference']==[1.,0.,0.]
    assert sample['preference']==[0.,1.,0.]
    assert sample['metrics']['preference']==preference.preference.as_dict()
    assert sample['metrics']['preference_quality_score']==terminal_quality_score(
        metrics_flow_objective(metrics,config),metrics['reconfiguration_cost'],metrics_workload_objective(metrics,config),
        config,preference=preference.preference)
