# utils_refactored.py
import os
import time
import json
import csv
import random
import itertools
import asyncio
import subprocess
import multiprocessing
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import torch
import memory_profiler
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from skopt import Optimizer
from skopt.space import Categorical, Real

# ---- Metric Logger ----
class MemoryEfficientMetricLogger:
    def __init__(self, log_dir: str = 'logs', log_file: str = 'metrics.csv', max_rows: int = 10000):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, log_file)
        self.max_rows = max_rows
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.metrics: List[Dict[str, Any]] = []
        self._last_save = time.time()
        self._save_interval = 30
        if os.path.exists(self.log_path):
            try:
                df = pd.read_csv(self.log_path)
                self.metrics = df.to_dict('records')
            except:
                self.metrics = []

    async def log(self, metrics: Dict[str, Any]) -> None:
        metrics['timestamp'] = datetime.utcnow().isoformat()
        self.metrics.append(metrics)
        self._cleanup()
        now = time.time()
        if now - self._last_save > self._save_interval:
            await asyncio.get_event_loop().run_in_executor(self._executor, self._save)
            self._last_save = now

    def _save(self) -> None:
        df = pd.DataFrame(self.metrics)
        if len(df) > self.max_rows:
            df = df.iloc[-self.max_rows:]
        df.to_csv(self.log_path, index=False)
        self.metrics = df.to_dict('records')
        self._cleanup()

    def _cleanup(self) -> None:
        gc = memory_profiler.memory_usage()
        torch.cuda.empty_cache()

# ---- Curriculum Manager ----
class MemoryEfficientCurriculumManager:
    def __init__(self, max_history: int = 1000, promote_threshold: int = 3):
        self.task_types = ['easy', 'medium', 'hard']
        self.current = 0
        self.history: List[str] = []
        self.max_history = max_history
        self.success = [0, 0, 0]
        self.attempts = [0, 0, 0]
        self.promote_threshold = promote_threshold

    def sample_task(self) -> str:
        choice = random.choice(self.task_types[:self.current + 1])
        self._record(choice)
        return choice

    def _record(self, task: str) -> None:
        self.history.append(task)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def update(self, result: Dict[str, Any]) -> None:
        idx = self.task_types.index(result.get('task_type', 'easy'))
        self.attempts[idx] += 1
        if result.get('success', False):
            self.success[idx] += 1
            if self.success[idx] >= self.promote_threshold and self.current < len(self.task_types)-1:
                self.current += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            'success': self.success.copy(),
            'attempts': self.attempts.copy(),
            'current': self.task_types[self.current],
            'recent': self.history[-10:]
        }

# ---- Meta Tuner ----
class MemoryEfficientMetaTuner:
    def __init__(self, mode: str = 'grid', max_configs: int = 1000):
        self.personas = ['INTP','ENFJ','ISTP','ENTJ']
        self.lrs = [1e-3,5e-4,1e-4]
        self.max_configs = max_configs
        self.history: List[Tuple[Dict[str,Any], float]] = []
        self.best = (None, float('-inf'))
        self.mode = mode
        self._tried = set()
        self._grid = list(itertools.product(self.personas, self.lrs))[:max_configs]
        if mode=='bayes':
            space = [Categorical(self.personas, name='persona'), Real(self.lrs[-1], self.lrs[0], prior='log-uniform', name='lr')]
            self.opt = Optimizer(space, random_state=42)
            self._asks: List[List[Any]] = []
        else:
            self.opt = None

    def suggest(self) -> Optional[Dict[str,Any]]:
        if len(self.history) >= self.max_configs:
            return None
        if self.mode=='grid':
            for p in self._grid:
                if p not in self._tried:
                    self._tried.add(p)
                    return {'persona':p[0],'lr':p[1]}
        elif self.mode=='random':
            p = (random.choice(self.personas), random.choice(self.lrs))
            if p not in self._tried:
                self._tried.add(p)
                return {'persona':p[0],'lr':p[1]}
        elif self.mode=='bayes' and self.opt:
            ask = self.opt.ask()
            self._asks.append(ask)
            return {'persona':ask[0],'lr':ask[1]}
        return None

    def update(self, cfg: Dict[str,Any], reward: float) -> None:
        self.history.append((cfg, reward))
        if reward > self.best[1]:
            self.best = (cfg, reward)
        if self.mode=='bayes' and self.opt:
            params = self._asks.pop(0)
            self.opt.tell(params, -reward)
        if len(self.history)>self.max_configs:
            self.history = self.history[-self.max_configs:]
            self._tried = {tuple(d.values()) for d,_ in self.history}

# ---- UI Tracker ----
class MemoryEfficientUITracker:
    def __init__(self, threshold: float = 0.05, max_frames: int = 100):
        self.threshold = threshold
        self.max_frames = max_frames
        self.buffer: List[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self.buffer.append(frame)
        if len(self.buffer)>self.max_frames:
            self.buffer = self.buffer[-self.max_frames:]

    def diff_score(self, f1: np.ndarray, f2: np.ndarray) -> float:
        return float(np.mean(np.abs(f1.astype(float)-f2.astype(float)))/255.0)

    def has_changed(self, f1: np.ndarray, f2: np.ndarray) -> bool:
        return self.diff_score(f1,f2)>self.threshold

    def change_mask(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        diff = np.abs(f1.astype(float)-f2.astype(float))
        mask = (diff>(self.threshold*255)).astype(np.uint8)*255
        return mask

# ---- Experiment Runner ----
class MemoryEfficientExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.tuner = MemoryEfficientMetaTuner(mode=args.tuner_mode)
        self.procs: List[Tuple[subprocess.Popen,Dict[str,Any]]] = []
        self.results: List[Dict[str,Any]] = []
        self.max_conc = min(args.num_experiments, multiprocessing.cpu_count())

    def run(self) -> None:
        os.makedirs(self.args.log_dir, exist_ok=True)
        for _ in range(self.args.num_experiments):
            while len(self.procs)>=self.max_conc:
                self._poll()
            cfg = self.tuner.suggest()
            if cfg is None:
                break
            log = os.path.join(self.args.log_dir, f"persona_{cfg['persona']}_lr_{cfg['lr']}_{int(time.time())}.csv")
            cmd = [sys.executable, self.args.script, '--persona', cfg['persona'], '--lr', str(cfg['lr']), '--log', log]
            p = subprocess.Popen(cmd)
            self.procs.append((p, {'config':cfg, 'log':log}))
        while self.procs:
            self._poll()
        self._aggregate()

    def _poll(self) -> None:
        time.sleep(1)
        for p,meta in self.procs[:]:
            if p.poll() is not None:
                df = pd.read_csv(meta['log'])
                reward = df['reward'].sum() if 'reward' in df else 0
                self.tuner.update(meta['config'], reward)
                self.results.append({'config':meta['config'],'reward':reward})
                self.procs.remove((p,meta))

    def _aggregate(self) -> None:
        aggregate_logs(self.args.log_dir)

# ---- Aggregation ----
def aggregate_logs(log_dir: str) -> None:
    files = glob(os.path.join(log_dir, '*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed = int(os.path.basename(f).split('_')[-1].split('.')[0])
        persona = os.path.basename(f).split('_')[1]
        df['seed'] = seed
        df['persona'] = persona
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    os.makedirs('analysis', exist_ok=True)
    stats = all_df.groupby('persona')['reward'].agg(['mean','std','max','min','count'])
    stats.to_csv('analysis/persona_stats.csv')
    plt.figure()
    for (_,grp) in all_df.groupby(['persona','seed']):
        plt.plot(grp['step'], grp['reward'], alpha=0.3)
    plt.savefig('analysis/all_runs.png')
    plt.close()

# ---- Evaluation ----
def kl_divergence(p: List[float], q: List[float], eps: float = 1e-8) -> float:
    return entropy(np.asarray(p)+eps, np.asarray(q)+eps)

def compute_win_rate(df: pd.DataFrame, col: str = 'total_reward', thresh: float = 0.1) -> float:
    c = col if col in df else 'reward'
    return float((df[c]>thresh).mean())

def compute_replay_speed(df: pd.DataFrame, time_col: str='timestamp', step_col: str='step') -> float:
    try:
        times = pd.to_datetime(df[time_col])
    except:
        times = pd.to_datetime(df[time_col], errors='coerce')
    secs = (times.iloc[-1]-times.iloc[0]).total_seconds()
    steps = df[step_col].iloc[-1]-df[step_col].iloc[0]+1
    return float(steps/secs) if secs>0 else float('nan')

def evaluate_log(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    return {'win_rate': compute_win_rate(df), 'speed': compute_replay_speed(df)}

# ---- Test Pipeline ----
def test_agent_pipeline(log_dir: str='logs/exp_runs', min_wr: float=0.1, min_sp: float=0.05) -> bool:
    files = glob(os.path.join(log_dir, '*.csv'))
    for f in files:
        m = evaluate_log(f)
        if m['win_rate']<min_wr or m['speed']<min_sp:
            return False
    return True
