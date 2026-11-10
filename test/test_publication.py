"""Publication must retain recomputable evidence while rejecting altered inputs."""
import json
import pytest
from scripts.publish_results import compact, digest, portable, verify_assets

def test_worse_sample_cannot_be_reported_as_best():
    from scripts.publish_results import validate_selection
    raw=[{'success':True,'quality':q,'sampling_repeat':i,'objectives':[q,2,3]} for i,q in enumerate((.2,.1,.3))]
    u={'arm':'full','deliveries':[{'success':True,'quality':.1,'objectives':[.1,2,3]}],
       'requests':[{'raw':raw,'selected_repeat':1}]}
    validate_selection(u)
    u['requests'][0]['selected_repeat']=0
    with pytest.raises(ValueError,match='selected candidate'):validate_selection(u)

def test_compaction_retains_numerical_candidates_and_redacts_paths():
    u={'arm':'full','deliveries':[{'success':True,'objectives':[1,2,3]}],
       'requests':[{'selected_repeat':0,'raw':[{'success':True,'objectives':[1,2,3],'actions':[4,5],'trace':[1]}]}],
       'model_timing':{'checkpoint':r'C:\Users\researcher\project\result\run\best_checkpoint.pt'}}
    u['content_sha256']=digest(u)
    c=compact(u)
    assert c['requests'][0]['raw']==[{'success':True,'objectives':[1,2,3]}]
    assert c['deliveries']==u['deliveries']
    assert c['original_content_sha256']==u['content_sha256']
    assert c['model_timing']['checkpoint']=='<WORKSPACE>/result/run/best_checkpoint.pt'
    assert c['content_sha256']==digest({k:v for k,v in c.items() if k!='content_sha256'})

def test_rule_and_search_evidence_survives_compaction():
    u={'raw_rule':{'success':True,'objectives':[1,2,3],'actions':[1]},
       'searches':[{'archive':[{'objectives':[1,2,3],'actions':[1],'action_trace_sha256':'abc'}],
                    'snapshots':[{'seconds':600,'paid_hv':.5}], 'candidate_history':[1]}],
       'content_sha256':'original'}
    c=compact(u)
    assert c['raw_rule']=={'success':True,'objectives':[1,2,3]}
    assert c['searches'][0]['archive']==[{'objectives':[1,2,3],'action_trace_sha256':'abc'}]
    assert c['searches'][0]['snapshots']==u['searches'][0]['snapshots']
    assert 'candidate_history' not in c['searches'][0]

def test_publication_blocks_credentials():
    with pytest.raises(ValueError,match='credential'):portable({'x':'ghp_'+'a'*40})

def test_asset_tampering_is_rejected(tmp_path,monkeypatch):
    import scripts.publish_results as p
    monkeypatch.setattr(p,'OUT',tmp_path)
    a=tmp_path/'part.zip';a.write_bytes(b'original')
    (tmp_path/'assets.json').write_text(json.dumps({'assets':[{'name':a.name,'bytes':8,'sha256':p.sha(a)}]}))
    verify_assets(tmp_path)
    a.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='asset differs'):verify_assets(tmp_path)
