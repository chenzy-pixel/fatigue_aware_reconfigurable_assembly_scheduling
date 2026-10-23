"""Stable explicit factorial validation subsets."""
from data.dataset import canonical_json_bytes, sha256_bytes
from data.distribution import stable_seed

def resolve_instance_indices(dataset, *, instance_indices=None, instance_limit=None, instance_offset=None):
    if instance_indices is not None:
        if instance_limit is not None or instance_offset is not None:
            raise ValueError('indices and offset/limit are exclusive')
        result = list(instance_indices)
    else:
        start = instance_offset or 0
        result = list(range(start, min(len(dataset), start + (instance_limit if instance_limit is not None else len(dataset)))))
    if not result or len(set(result)) != len(result) or any((i < 0 or i >= len(dataset) for i in result)):
        raise ValueError('invalid dataset indices')
    return result

def subset_snapshot(dataset, indices, *, role):
    p = {'selection_version': 'factorial_v1', 'dataset_manifest_sha256': sha256_bytes(canonical_json_bytes(dataset.manifest)), 'instance_indices': list(indices), 'files': [dataset.manifest['files'][i] for i in indices]}
    return {**p, 'role': role, 'subset_sha256': sha256_bytes(canonical_json_bytes(p))}

def select_validation_subsets(dataset, weights=None, *, target_count=27, diagnostic_count=0):
    if target_count + diagnostic_count > len(dataset):
        raise ValueError('insufficient independent validation instances')
    indices = list(range(len(dataset)))
    result = {'target': subset_snapshot(dataset, indices[:target_count], role='target')}
    if diagnostic_count:
        result['diagnostic'] = subset_snapshot(dataset, indices[target_count:target_count + diagnostic_count], role='diagnostic')
    return result
