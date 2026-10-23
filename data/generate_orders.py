"""Generate explicit factorial cells, independently of solver performance."""
import random
from data.models import AssemblyInstance, FatigueSpec, MachineSpec, MachineModuleSpec, WorkerSpec, OrderSpec, OperationSpec, ModuleCostSpec
from data.distribution import CELLS, FactorCell, stable_seed, protocol_hashes
from data.feasibility import cheap_feasibility_precheck

def generate_instance(config, cell: FactorCell, seed: int, *, split='train'):
    g = config['generator']
    rng = random.Random(seed)
    m, w, a = (cell.M, cell.W, cell.A)
    machine_ids = [f'M{i + 1}' for i in range(m)]
    rng.shuffle(machine_ids)
    module_ids = [f'A{i + 1}' for i in range(a)]
    rng.shuffle(module_ids)
    worker_ids = [f'H{i + 1}' for i in range(w)]
    rng.shuffle(worker_ids)
    edges = [tuple(sorted((i, (i + 1) % m))) for i in range(m)] + [(i, i + m // 2) for i in range(m // 2)]
    u = lambda name: rng.uniform(*g[name])
    base = {x: (u('disassembly_time'), u('installation_time')) for x in module_ids}
    machines = tuple((MachineSpec(machine_ids[i], module_ids[i], u('downtime_cost'), {x: MachineModuleSpec(base[x][1], base[x][0], u('processing_factor')) for j, x in enumerate(module_ids) if i in edges[j]}) for i in range(m)))
    workers = tuple((WorkerSpec(worker_ids[i], tuple((module_ids[(2 * i + k) % a] for k in range(4))), u('labor_cost'), u('initial_fatigue')) for i in range(w)))
    demand = [i % a for i in range(cell.N * cell.K)]
    for _ in range(10 * len(demand)):
        i, j = (rng.randrange(len(demand)), rng.randrange(len(demand)))
        demand[i], demand[j] = (demand[j], demand[i])
        if any((len(set(demand[t * cell.K:(t + 1) * cell.K])) < 2 for t in {i // cell.K, j // cell.K})):
            demand[i], demand[j] = (demand[j], demand[i])
    orders = []
    for j in range(cell.N):
        route = demand[j * cell.K:(j + 1) * cell.K]
        rng.shuffle(route)
        jid = f'J{j + 1}'
        orders.append(OrderSpec(jid, 0.0, tuple((OperationSpec(f'O{j + 1}_{k + 1}', jid, k + 1, module_ids[v], float(rng.randint(*g['processing_time']))) for k, v in enumerate(route)))))
    instance = AssemblyInstance('3.0.0', f'{split}_{cell.cell_id}_{seed}', 'factorial_batch', g['resolution'], tuple(module_ids), 'A0', machines, workers, tuple(orders), FatigueSpec(**g['fatigue']), {x: ModuleCostSpec(float(rng.randint(*g['fixed_disassembly_cost'])), float(rng.randint(*g['fixed_installation_cost']))) for x in module_ids}, 0.0)
    report = cheap_feasibility_precheck(instance)
    if not report.passed:
        raise ValueError(report.to_dict())
    return instance

class InstanceGenerator:

    def __init__(self, config):
        self.config = config

    def generate(self, *, seed, cell, split='train'):
        from data.dataset import GeneratedInstanceRecord
        from dataclasses import asdict
        instance = generate_instance(self.config, cell, seed, split=split)
        return GeneratedInstanceRecord(instance, {'seed': seed, 'split': split, 'cell_id': cell.cell_id, 'scale': cell.scale, **asdict(cell), 'W': cell.W, 'A': cell.A, 'generator_version': '3.0.0', 'feasibility_status': 'unknown', **protocol_hashes(self.config)})
