import os
import json
import random
import argparse
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import ray
from dataclasses import dataclass, asdict

# Import your modules
from models import MemoryEfficientVisionModel, ReplayBuffer, create_vision_model
from faiss_memory import FAISSMemory
from logic import EpisodeRunner
from utils import MemoryEfficientMetricLogger as MetricLogger

# ----------------------------------------
# Genetic Algorithm Entities
# ----------------------------------------
@dataclass
class AgentGenes:
    exploration_bias: float
    risk_tolerance: float
    learning_rate: float
    memory_decay: float
    mutation_rate: float
    subgoal_weights: List[float]
    pattern_weight: float
    adaptation_speed: float

class EvolutionManager:
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_fraction: float = 0.1,
        selection_pressure: float = 2.0,
        initial_genes: Dict[str, Any] = None,
        gene_bounds: Dict[str, Tuple[float, float]] = None,
        checkpoint_path: str = 'evo_population.json'
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.selection_pressure = selection_pressure
        self.gene_bounds = gene_bounds or {
            'exploration_bias': (0.0, 1.0),
            'risk_tolerance': (0.0, 1.0),
            'learning_rate': (1e-5, 1e-2),
            'memory_decay': (0.0, 0.1),
            'mutation_rate': (0.01, 0.2),
            'subgoal_weights': ( [0.0,0.0,0.0,0.0], [1.0,1.0,1.0,1.0] ),
            'pattern_weight': (0.1, 0.9),
            'adaptation_speed': (0.1, 0.5)
        }
        self.checkpoint_path = checkpoint_path
        self.population: List[Dict[str, Any]] = []
        self.fitness: Dict[int, float] = {}
        self.generation = 0
        self._init_population(initial_genes)

    def _init_population(self, initial_genes):
        base = self._random_genes() if initial_genes is None else AgentGenes(**initial_genes)
        for i in range(self.population_size):
            genes = base if i < int(self.elite_fraction*self.population_size) else self._random_genes()
            self.population.append({'id': i, 'genes': genes, 'fitness': 0.0})

    def _random_genes(self) -> AgentGenes:
        genes = {}
        for k, bounds in self.gene_bounds.items():
            if k == 'subgoal_weights':
                lows, highs = bounds
                genes[k] = [random.uniform(l, h) for l, h in zip(lows, highs)]
            else:
                low, high = bounds
                genes[k] = random.uniform(low, high)
        return AgentGenes(**genes)

    def update_fitness(self, agent_id: int, fitness: float):
        self.fitness[agent_id] = fitness
        for member in self.population:
            if member['id'] == agent_id:
                member['fitness'] = fitness
                break

    def evolve(self):
        # Sort and select elites
        ranked = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        elites = ranked[:int(self.elite_fraction*self.population_size)]
        new_pop = [ {'id': e['id'], 'genes': e['genes'], 'fitness': 0.0} for e in elites ]
        # Generate offspring
        while len(new_pop) < self.population_size:
            p1, p2 = self._tournament_select(), self._tournament_select()
            child = self._crossover(p1['genes'], p2['genes'])
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            new_id = len(new_pop)
            new_pop.append({'id': new_id, 'genes': child, 'fitness': 0.0})
        self.population = new_pop
        self.fitness.clear()
        self.generation += 1
        self.save()

    def _tournament_select(self) -> Dict[str, Any]:
        contenders = random.sample(self.population, k=int(self.selection_pressure))
        return max(contenders, key=lambda x: x['fitness'])

    def _crossover(self, g1: AgentGenes, g2: AgentGenes) -> AgentGenes:
        d1, d2 = asdict(g1), asdict(g2)
        child = {}
        for k in d1:
            if k=='subgoal_weights':
                child[k] = [random.choice([a,b]) for a,b in zip(d1[k], d2[k])]
            else:
                alpha = random.random()
                child[k] = d1[k]*alpha + d2[k]*(1-alpha)
        return AgentGenes(**child)

    def _mutate(self, genes: AgentGenes) -> AgentGenes:
        d = asdict(genes)
        for k,bounds in self.gene_bounds.items():
            if k=='subgoal_weights':
                lows, highs = bounds
                d[k] = [ np.clip(v + np.random.normal(scale=0.1), low, high)
                         for v,low,high in zip(d[k], lows, highs)]
            else:
                low, high = bounds
                d[k] = float(np.clip(d[k] + np.random.normal(scale=0.1), low, high))
        return AgentGenes(**d)

    def save(self):
        data = {'generation': self.generation,
                'population': [ {'id':m['id'], 'genes': asdict(m['genes']), 'fitness':m['fitness']} for m in self.population ]}
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.checkpoint_path): return
        data = json.load(open(self.checkpoint_path))
        self.generation = data['generation']
        self.population = []
        for m in data['population']:
            self.population.append({'id':m['id'], 'genes':AgentGenes(**m['genes']), 'fitness':m['fitness']})
        self.fitness = {m['id']:m['fitness'] for m in self.population}

# ----------------------------------------
# Ray evaluation worker
# ----------------------------------------
@ray.remote
class EvalWorker:
    def __init__(self, config: dict, episodes: int):
        self.config = config
        self.episodes = episodes

    def evaluate(self, member: Dict[str, Any]) -> Tuple[int, float]:
        try:
            gene_cfg = asdict(member['genes'])
            runner_cfg = {**self.config, **gene_cfg}
            total = 0.0
            for _ in range(self.episodes):
                runner = EpisodeRunner(runner_cfg)
                avg = runner.run_episode_sync()
                total += avg
            return member['id'], total/self.episodes
        except Exception:
            traceback.print_exc()
            return member['id'], float('-inf')

# ----------------------------------------
# Controller
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='evo_config.json')
    args = parser.parse_args()
    # Load controller config
    cfg = json.load(open(args.config))
    evo_cfg = cfg['evolution']
    runner_cfg = cfg['runner']
    episodes = cfg.get('episodes', 3)
    generations = cfg.get('generations', 10)

    ray.init(ignore_reinit_error=True)
    manager = EvolutionManager(**evo_cfg)

    for gen in range(generations):
        print(f"=== Generation {gen}, pop size {manager.population_size} ===")
        # Evaluate population
        futures = [EvalWorker.remote(runner_cfg, episodes).evaluate.remote(m)
                   for m in manager.population]
        results = ray.get(futures)
        for aid, fit in results:
            manager.update_fitness(aid, fit)
        # Logging average and best
        fits = [m['fitness'] for m in manager.population]
        logger = MetricLogger(log_dir=runner_cfg['log_dir'], log_file="evo_fitness.csv")
        logger.log({'generation': gen, 'avg_fitness': np.mean(fits), 'best_fitness': max(fits)})
        # Evolve
        manager.evolve()
    ray.shutdown()

if __name__ == '__main__':
    main()
