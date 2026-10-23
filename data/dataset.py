"""Explicit sealed manifests and independent online factorial instances."""
import hashlib, json
from pathlib import Path
from dataclasses import dataclass
from data.models import instance_to_dict, parse_instance_dict
from data.distribution import training_sampling_plan, stable_seed, protocol_hashes
PERSISTED_SPLITS = ('calibration', 'validation', 'test')
ALL_SPLITS = ('train', *PERSISTED_SPLITS)

def canonical_json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())

def template_sha256(instance):
    return sha256_bytes(canonical_json_bytes(instance_to_dict(instance)))

def validate_algorithm_seed(config, seed):
    return int(seed)

@dataclass(frozen=True)
class GeneratedInstanceRecord:
    instance: object
    metadata: dict

    def to_dict(self):
        return {'instance': instance_to_dict(self.instance), 'metadata': self.metadata}

def save_generated_record_atomic(record, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record.to_dict())
    tmp = p.with_suffix('.tmp')
    tmp.write_bytes(payload)
    tmp.replace(p)
    return sha256_bytes(payload)

def load_generated_record(path, *, expected_sha256=None):
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise ValueError('instance hash mismatch')
    raw = json.loads(payload)
    return GeneratedInstanceRecord(parse_instance_dict(raw['instance']), raw['metadata'])

class InstanceDataset:

    def __init__(self, manifest_path, instances_root, config):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.instances_root = Path(instances_root)
        if self.manifest.get('schema_version') != '3.0.0':
            raise ValueError('dataset schema 3 required')
        for key, value in protocol_hashes(config).items():
            if self.manifest.get(key) != value:
                raise ValueError('dataset protocol mismatch: ' + key)

    def __len__(self):
        return len(self.manifest['files'])

    def __getitem__(self, index):
        row = self.manifest['files'][index]
        p = (self.instances_root / row['path']).resolve()
        if not p.is_relative_to(self.instances_root.resolve()):
            raise ValueError('unsafe instance path')
        return load_generated_record(p, expected_sha256=row['sha256'])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class OnlineInstanceDataset:

    def __init__(self, *, config, episode_count, template=None):
        from data.generate_orders import InstanceGenerator
        self.config = config
        self.episode_count = episode_count
        self.seed_start = 1000000
        self.generator = InstanceGenerator(config)
        self.sampling_plan = training_sampling_plan(config, episode_count)

    def __len__(self):
        return self.episode_count

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.generator.generate(seed=stable_seed('factorial27_v1', 'train', index), cell=self.sampling_plan[index])

    def get_with_cache_info(self, index):
        from configs import project_path
        if not 0 <= index < len(self):
            raise IndexError(index)
        fingerprint = sha256_bytes(canonical_json_bytes(protocol_hashes(self.config)))
        path = project_path(self.config['paths']['training_instances_cache']) / fingerprint / f'{index}.json'
        if path.is_file():
            record = load_generated_record(path)
            # The cache is reproducibility evidence, not an independent data
            # source. Verify it against the deterministic generator, including
            # same-cell corruption that structural validation cannot detect.
            expected = self[index]
            expected.metadata['training_cache_fingerprint'] = fingerprint
            if canonical_json_bytes(record.to_dict()) != canonical_json_bytes(expected.to_dict()):
                raise ValueError(f'training cache differs from deterministic instance: {path}')
            return (record, True, sha256_file(path), path)
        record = self[index]
        record.metadata['training_cache_fingerprint'] = fingerprint
        digest = save_generated_record_atomic(record, path)
        return (record, False, digest, path)

def load_dataset_split(config, split, **kwargs):
    from configs import project_path
    if split not in PERSISTED_SPLITS:
        raise ValueError('unknown batch dataset split')
    return InstanceDataset(project_path(config['paths']['manifests_root']) / split / 'manifest.json', project_path(config['paths']['instances_root']), config)
