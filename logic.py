import os
import sys
import time
import json
import torch
import ray
import argparse
import traceback
import asyncio
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
import cv2
import pytesseract
from torchvision import transforms

from agent import DQNAgent, PPOAgent
from faiss_memory import FAISSMemory
from models import (
    IntrinsicCuriosityModule,
    ShortHorizonPlanner,
    ReplayBuffer,
    create_vision_model
)
from data import (
    AsyncActionExecutor as ActionExecutor,
    capture_single_screenshot,
    XMLParser,
    LazyEventParser,
    init_dirs
)
from utils import (
    MemoryEfficientMetricLogger as MetricLogger,
    MemoryEfficientCurriculumManager as CurriculumManager,
    MemoryEfficientMetaTuner as MetaTuner,
    MemoryEfficientUITracker as UITracker
)

# ----------------------------------------------------
# Checkpoint Manager
# ----------------------------------------------------
class CheckpointManager:
    def __init__(self, ckpt_dir: str = 'checkpoints'):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, step: int, state: dict):
        path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pt")
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_latest(self):
        files = [f for f in os.listdir(self.ckpt_dir) if f.endswith('.pt')]
        if not files:
            return None
        latest = sorted(files)[-1]
        path = os.path.join(self.ckpt_dir, latest)
        print(f"Loading checkpoint: {path}")
        return torch.load(path)

# ----------------------------------------------------
# Episode Runner with XML & Event Parsing
# ----------------------------------------------------
class EpisodeRunner:
    def __init__(self, config: dict):
        self.config = config
        self.max_steps = config.get('max_steps', 1000)
        self.device = torch.device(config.get('device', 'cpu'))
        self.save_interval = config.get('save_interval', 500)

        # Prepare directories
        init_dirs()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # Initialize modules
        self._init_modules(config)
        self.ckpt_mgr = CheckpointManager(config.get('ckpt_dir', 'checkpoints'))

    def _init_modules(self, config):
        # Vision model
        self.vision_model = create_vision_model().to(self.device).eval()
        # Agents
        lr = config.get('lr', None)
        self.dqn = DQNAgent(device=self.device, lr=lr)
        self.ppo = PPOAgent(input_dim=1280+2, num_subgoals=4, device=self.device, lr=lr)
        # Intrinsic Curiosity
        self.icm = IntrinsicCuriosityModule(feature_dim=1280, action_dim=2).to(self.device)
        # Planner
        self.planner = ShortHorizonPlanner(max_lookahead=3)
        # Replay & Memory
        self.buffer = ReplayBuffer(capacity=config.get('buffer_capacity', 10000))
        self.faiss = FAISSMemory(embedding_dim=1280, max_size=10000, use_cosine=True)
        # Executors & utilities
        self.executor = ActionExecutor()
        self.logger = MetricLogger(log_dir=config['log_dir'], log_file=config['log_file'])
        self.curriculum = CurriculumManager()
        self.tuner = MetaTuner()
        self.tracker = UITracker()

    async def run_episode(self, start_step: int = 0):
        step = start_step
        prev_feat = None
        prev_action = None
        prev_img = None
        prev_ocr = None

        # Initialize event parsers
        event_dir = 'collected_data/events'
        xml_dir = 'collected_data/xml'
        parsers = []

        while step < self.max_steps:
            try:
                # 1) Capture screenshot
                png = capture_single_screenshot()
                img = Image.open(BytesIO(png)).convert('RGB')
                tensor_img = self.transform(img).unsqueeze(0).to(self.device)

                # 2) Extract vision features
                with torch.no_grad():
                    feats_map = self.vision_model.features(tensor_img)
                    pooled = feats_map.mean(dim=[2,3]).squeeze(0)
                feat_vec = pooled.cpu().numpy()

                # 3) Parse latest XML state
                xml_files = sorted(os.listdir(xml_dir))
                num_clickable, avg_size = 0, 0
                if xml_files:
                    parser = XMLParser(os.path.join(xml_dir, xml_files[-1]))
                    ui_feats = []
                    async for elem in parser.parse():
                        ui_feats.append((elem['clickable'], elem['bounds'][2]-elem['bounds'][0], elem['bounds'][3]-elem['bounds'][1]))
                    num_clickable = sum(1 for f in ui_feats if f[0])
                    sizes = [f[1]*f[2] for f in ui_feats]
                    avg_size = float(sum(sizes)/len(sizes)) if sizes else 0.0

                # Concatenate UI summary
                state_feat = np.concatenate([feat_vec, [num_clickable, avg_size]])

                # 4) Select actions
                dqn_act = self.dqn.select_action(state_feat)
                ppo_act, _, _, _ = self.ppo.select_action(state_feat)

                # 5) Execute DQN tap
                await self.executor.tap_norm(*dqn_act)

                # 6) Compute reward
                curr_img_arr = np.array(img, dtype=np.float32)
                if prev_img is not None:
                    prev_arr = np.array(Image.open(BytesIO(prev_img)).convert('RGB'), dtype=np.float32)
                    img_diff = float(np.abs(prev_arr - curr_img_arr).mean()/255.0)
                    reward = -img_diff
                    text = pytesseract.image_to_string(img)
                    nums = [int(n) for n in json.loads(json.dumps(re.findall(r'\d+', text)))] if text else []
                    ocr_score = max(nums) if nums else 0
                    if prev_ocr is not None:
                        reward += 0.3*(ocr_score - prev_ocr)
                    prev_ocr = ocr_score
                else:
                    reward, img_diff, ocr_score = 0.0, 0.0, 0

                # Intrinsic reward
                if prev_feat is not None and prev_action is not None:
                    act_t = torch.tensor(prev_action, dtype=torch.float32, device=self.device).unsqueeze(0)
                    intr = self.icm.compute_intrinsic_reward(
                        torch.tensor(prev_feat).unsqueeze(0).to(self.device),
                        pooled.unsqueeze(0),
                        act_t
                    )[0]
                else:
                    intr = 0.0
                total_reward = reward + float(intr)

                # 7) Store and optimize
                if prev_feat is not None:
                    self.buffer.push(prev_feat, prev_action, total_reward, state_feat, False)
                self.ppo.buffer.append((state_feat, ppo_act, None, total_reward, False, None))
                self.faiss.add(np.array([feat_vec],dtype=np.float32), [{'step':step,'subgoal':ppo_act,'reward':total_reward}])

                if step % 5 == 0:
                    loss = self.dqn.optimize(self.buffer, batch_size=8)
                if step>0 and step%128==0:
                    pol_loss, val_loss = self.ppo.update()

                # 8) UI change detection
                if prev_img is not None:
                    f1 = prev_arr.astype(np.uint8)
                    f2 = curr_img_arr.astype(np.uint8)
                    ui_changed = self.tracker.has_changed(f1,f2)
                    ui_score = self.tracker.diff_score(f1,f2)
                else:
                    ui_changed, ui_score = False, 0.0

                # 9) Logging
                self.logger.log({
                    'timestamp':datetime.utcnow().isoformat(),
                    'step':step,
                    'dqn_action':dqn_act,
                    'ppo_subgoal':ppo_act,
                    'reward':reward,
                    'intrinsic':intr,
                    'total_reward':total_reward,
                    'img_diff':img_diff,
                    'ocr_score':ocr_score,
                    'ui_changed':ui_changed,
                    'ui_score':ui_score
                })

                # Checkpoint
                if step>0 and step%self.save_interval==0:
                    state = {'step':step,'dqn':self.dqn.state_dict(),'ppo':self.ppo.state_dict(),'buffer':list(self.buffer.buffer),'faiss_meta':self.faiss.meta}
                    self.ckpt_mgr.save(step,state)

                # Prepare next
                prev_feat = state_feat
                prev_action = dqn_act
                prev_img = png
                step+=1

            except Exception as e:
                print(f"[ERROR] Step {step} exception: {e}")
                traceback.print_exc()
                step +=1
                continue

        # Finish
        self.logger._save()
        print("Episode completed.")

# ----------------------------------------------------
# Distributed Worker
# ----------------------------------------------------
@ray.remote
class Worker:
    def __init__(self, config):
        self.runner = EpisodeRunner(config)

    def run(self, resume: bool = True):
        start = 0
        if resume:
            ckpt = self.runner.ckpt_mgr.load_latest()
            if ckpt:
                start = ckpt['step']
                self.runner.dqn.load_state_dict(ckpt['dqn'])
                self.runner.ppo.load_state_dict(ckpt['ppo'])
        asyncio.get_event_loop().run_until_complete(self.runner.run_episode(start))
        return True

# ----------------------------------------------------
# Orchestration
# ----------------------------------------------------
def distributed_train(config: dict, num_workers: int = 2):
    ray.init(ignore_reinit_error=True)
    workers = [Worker.remote(config) for _ in range(num_workers)]
    results = ray.get([w.run.remote() for w in workers])
    print(f"Distributed results: {results}")
    ray.shutdown()

# ----------------------------------------------------
# CLI
# ----------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cfg.setdefault('log_dir','logs')
    cfg.setdefault('log_file','agent_metrics.csv')

    if args.distributed:
        distributed_train(cfg, num_workers=args.workers)
    else:
        runner=EpisodeRunner(cfg)
        asyncio.get_event_loop().run_until_complete(runner.run_episode())
