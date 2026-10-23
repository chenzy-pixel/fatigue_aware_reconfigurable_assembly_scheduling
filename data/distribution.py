"""Versioned, prefix-stable factorial instance streams."""
import hashlib
import json
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class FactorCell:
    N: int
    K: int
    M: int

    @property
    def W(self):
        return 3 * self.M // 4

    @property
    def A(self):
        return 3 * self.M // 2

    @property
    def scale(self):
        return {12: 'small', 24: 'medium', 48: 'large'}[self.N]

    @property
    def cell_id(self):
        return f'C{1 + 9 * (12, 24, 48).index(self.N) + 3 * (4, 8, 12).index(self.M) + (3, 4, 5).index(self.K):02d}'
CELLS = tuple((FactorCell(n, k, m) for n in (12, 24, 48) for m in (4, 8, 12) for k in (3, 4, 5)))

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()[:8], 'big')

def benchmark_environment_config(config):
    return {'physics': 'batch_event_v1', 'fatigue_mode': 'full'}

def protocol_hashes(config, settings=None):
    validate_training_distribution(config)
    inputs = {'generator_config_sha256': settings or config['generator'], 'environment_config_sha256': benchmark_environment_config(config), 'distribution_contract_sha256': {'version': 'factorial27_v1', 'generator': settings or config['generator'], 'cells': [(c.N, c.K, c.M) for c in CELLS]}}
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in inputs.items()}

def validate_training_distribution(config):
    if config['generator'].get('version') != '3.0.0' or set(config['generator']) & {'pressure_profiles', 'dataset_pressure_weights', 'curriculum', 'severity_curriculum', 'wave_count'}:
        raise ValueError('generator schema 3 required; historical protocols use archived code')
    expected = {'N': [12, 24, 48], 'K': [3, 4, 5], 'M': [4, 8, 12]}
    if config.get('dataset', {}).get('factor_levels') != expected:
        raise ValueError('factor_levels must match the frozen factorial27_v1 contract; new levels require a new data protocol')

def training_sampling_plan(config, episode_count):
    validate_training_distribution(config)
    result = []
    for r in range((episode_count + 26) // 27):
        cells = list(CELLS)
        random.Random(stable_seed('factorial27_v1', config['generator'], r)).shuffle(cells)
        result.extend(cells)
    return result[:episode_count]
