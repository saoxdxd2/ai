import os
import sys
import time
import random
import gc
import argparse
import memory_profiler
import numpy as np
import torch
import json
import re
from datetime import datetime
from io import BytesIO
from PIL import Image
import cv2
import pytesseract

from agent import DQNAgent, PPOAgent
from faiss_memory import FAISSMemory
from models import create_vision_model, IntrinsicCuriosityModule, ShortHorizonPlanner, ReplayBuffer
from data import AsyncActionExecutor as ActionExecutor, get_screen_size as load_config, capture_single_screenshot
from utils import (
    MemoryEfficientMetricLogger as MetricLogger,
    MemoryEfficientCurriculumManager as CurriculumManager,
    MemoryEfficientMetaTuner as MetaTuner,
    MemoryEfficientUITracker as UITracker
)
from evolution_manager import EvolutionManager
from models.evolving_agent import EvolvingAgent

# --- Image preprocessing ---
def preprocess_image(png_bytes, transform):
    """Converts PNG bytes to a tensor suitable for the model."""
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

# --- Main training loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--persona', type=str, default='INTP')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--log_path', type=str, default='logs/agent_metrics.csv')
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load personas
    with open('persona.json') as f:
        personas = [p for p in json.load(f) if isinstance(p, dict) and 'mbti' in p]
    persona = next((p for p in personas if p['mbti'] == args.persona), personas[0] if personas else {})
    print(f"Loaded persona: {persona.get('mbti','UNKNOWN')}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vision model
    vision_model = create_vision_model()
    checkpoint = torch.load('models/vision_model.pth', map_location=device)
    vision_model.load_state_dict(checkpoint['model_state_dict'])
    vision_model.to(device).eval()

    # Image transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Action executor
    executor = ActionExecutor()

    # Initialize evolution manager
    evolution_manager = EvolutionManager(
        population_size=32,
        mutation_rate=0.1,
        selection_pressure=0.5,
        num_elites=4
    )
    
    # Create evolving agent
    agent = EvolvingAgent(
        agent_id=0,  # Start with agent 0
        evolution_manager=evolution_manager,
        vision_model=vision_model,
        memory=faiss_mem
    )
    
    # Agents and modules
    buffer = ReplayBuffer(capacity=10000)
    dqn_agent = DQNAgent(device=device, lr=args.lr) if args.lr else DQNAgent(device=device)
    ppo_agent = PPOAgent(input_dim=1280, num_subgoals=4, device=device, lr=args.lr) if args.lr else PPOAgent(input_dim=1280, num_subgoals=4, device=device)
    icm = IntrinsicCuriosityModule(feature_dim=1280, action_dim=2).to(device)
    planner = ShortHorizonPlanner(max_lookahead=3)
    
    # Utility managers
    logger = MetricLogger(log_dir=os.path.dirname(args.log_path), log_file=os.path.basename(args.log_path))
    curriculum = CurriculumManager()
    tuner = MetaTuner()
    tracker = UITracker()

    # State placeholders
    prev_img = None
    prev_state = None
    prev_action = None
    prev_ocr = None
    step = 0

    print("Starting training loop. Press Ctrl+C to stop.")
    try:
        while step < args.max_steps:
            # 1) Capture screenshot
            png = capture_single_screenshot()
            state_tensor = preprocess_image(png, transform).to(device)

            # 2) Extract vision features
            with torch.no_grad():
                feats = vision_model.features(state_tensor)
                pooled = feats.mean(dim=[2, 3]).squeeze(0).cpu().numpy()

            # 3) Get evolving agent action
            action = agent.get_action(state_tensor)
            x_rat, y_rat = action

            # 4) Execute tap
            executor.tap_norm(x_rat, y_rat)

            # 5) Compute reward
            img_arr = np.array(Image.open(BytesIO(png)).convert('RGB'))
            if prev_img is not None:
                prev_arr = np.array(Image.open(BytesIO(prev_img)).convert('RGB'))
                img_diff = float(np.abs(prev_arr - img_arr).mean() / 255.0)
                reward = -img_diff
                # OCR-based reward
                text = pytesseract.image_to_string(Image.fromarray(img_arr))
                nums = [int(n) for n in re.findall(r'\d+', text)] if text else []
                ocr_score = max(nums) if nums else 0
                if prev_ocr is not None:
                    reward += 0.3 * (ocr_score - prev_ocr)
                prev_ocr = ocr_score
            else:
                img_diff = 0.0
                prev_ocr = 0

            # 6) Intrinsic curiosity
            if prev_state is not None and prev_action is not None:
                prev_feat = prev_state.mean(dim=[2, 3]).squeeze(0)
                curr_feat = state_tensor.mean(dim=[2, 3]).squeeze(0)
                act_t = torch.tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
                intrinsic = icm.compute_intrinsic_reward(prev_feat.unsqueeze(0), curr_feat.unsqueeze(0), act_t)[0]
            else:
                intrinsic = 0.0
            total_reward = reward + float(intrinsic)

            # 7) Update evolving agent
            agent.update_fitness(total_reward)
            agent.remember(state_tensor, action, total_reward)

            # 8) Store transitions
            if prev_state is not None:
                buffer.push(prev_state.cpu().numpy(), prev_action, total_reward, state_tensor.cpu().numpy(), False)
            
            # 9) Optimize agents
            if step % 5 == 0:
                dqn_loss = dqn_agent.optimize(buffer, batch_size=8)
                if dqn_loss is not None:
                    print(f"[DQN] step={step}, loss={dqn_loss:.4f}")
            if step > 0 and step % 128 == 0:
                pl, vl = ppo_agent.update()
                print(f"[PPO] step={step}, policy_loss={pl:.4f}, value_loss={vl:.4f}")

            # 10) UI change detection
            if prev_img is not None:
                f1 = np.array(Image.open(BytesIO(prev_img)).convert('RGB'))
                f2 = img_arr
                ui_changed = tracker.has_changed(f1, f2)
                ui_score = tracker.diff_score(f1, f2)
                if ui_changed:
                    mask = tracker.change_mask(f1, f2)
                    cv2.imwrite(f"logs/ui_change_{step}.png", mask)
            else:
                ui_changed = False
                ui_score = 0.0

            # 11) Logging
            logger.log({
                'timestamp': datetime.utcnow().isoformat(),
                'step': step,
                'persona': persona.get('mbti', 'UNKNOWN'),
                'subgoal': agent.genes['subgoal_weights'],
                'tap': (x_rat, y_rat),
                'reward': reward,
                'intrinsic': intrinsic,
                'total_reward': total_reward,
                'dqn_loss': dqn_loss if 'dqn_loss' in locals() else None,
                'ppo_policy_loss': pl if 'pl' in locals() else None,
                'ppo_value_loss': vl if 'vl' in locals() else None,
                'ui_changed': ui_changed,
                'ui_score': ui_score,
                'generation': agent.state['generation'],
                'exploration': agent.genes['exploration_bias'],
                'risk': agent.genes['risk_tolerance']
            })

            # Update state
            prev_state = state_tensor
            prev_action = [x_rat, y_rat]
            prev_img = png
            step += 1
            time.sleep(1)

    except KeyboardInterrupt:
        logger._save()
        print("Agent stopped. Metrics saved.")

if __name__ == '__main__':
    main()
