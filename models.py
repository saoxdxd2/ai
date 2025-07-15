# models.py - Cleaned and Optimized Version
"""
Core models and utilities for vision-based RL system.
Includes memory-efficient implementations with proper resource management.
"""

import os
import json
import time
import gc
import psutil
import pickle
import re
from collections import deque, namedtuple
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from io import BytesIO
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import cv2

# Vision and ML imports
from torchvision import models, transforms
import pytesseract

# Distributed computing
import ray
import faiss

# Data processing
import pandas as pd
from glob import glob

# Web automation
import asyncio
from playwright.async_api import async_playwright

# Constants
PARSED_DIR = "parsed_data"
IMG_DIR = os.path.join("collected_data", "screenshots")
FAISS_AVAILABLE = True

try:
    import faiss
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using fallback memory implementation")


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

class MemoryManager:
    """Centralized memory management utility"""
    
    def __init__(self, gc_threshold=0.8, cleanup_interval=30):
        self.gc_threshold = gc_threshold
        self.cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        current_time = time.time()
        return current_time - self._last_cleanup > self.cleanup_interval
    
    def cleanup(self, force=False):
        """Perform memory cleanup"""
        if not force and not self.should_cleanup():
            return
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check memory usage
        process = psutil.Process()
        mem_info = process.memory_info()
        if mem_info.rss / psutil.virtual_memory().total > self.gc_threshold:
            # Force aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self._last_cleanup = time.time()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(force=True)


# ============================================================================
# DATASET IMPLEMENTATIONS
# ============================================================================

class MemoryEfficientImageDataset(Dataset):
    """Memory-efficient lazy loading image dataset with caching"""
    
    def __init__(self, img_dir: str, transform=None, max_cache_size=100):
        self.img_dir = img_dir
        self.transform = transform
        self.max_cache_size = max_cache_size
        self.memory_manager = MemoryManager()
        
        # Cache management
        self._cache = {}
        self._cache_order = []
        
        # Load image paths
        self._load_paths()
    
    def _load_paths(self):
        """Load and sort image paths"""
        self.img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(root, file))
        self.img_paths.sort()
    
    def _manage_cache(self, img_path: str, img: Image.Image):
        """Manage cache size and add new image"""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest item
            oldest_path = self._cache_order.pop(0)
            if oldest_path in self._cache:
                old_img = self._cache.pop(oldest_path)
                if hasattr(old_img, 'close'):
                    old_img.close()
        
        self._cache[img_path] = img
        self._cache_order.append(img_path)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Memory cleanup check
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup()
        
        # Check cache first
        if img_path in self._cache:
            img = self._cache[img_path]
        else:
            # Load image
            with Image.open(img_path) as loaded_img:
                img = loaded_img.convert('RGB').copy()
            
            # Cache management
            self._manage_cache(img_path, img)
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, os.path.basename(img_path)
    
    def __del__(self):
        """Cleanup resources"""
        for img in self._cache.values():
            if hasattr(img, 'close'):
                img.close()
        self._cache.clear()
        self._cache_order.clear()


class TapActionDataset(Dataset):
    """Dataset for loading screenshots and corresponding tap actions"""
    
    def __init__(self, parsed_dir: str, img_dir: str, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load gesture data
        for json_file in sorted(os.listdir(parsed_dir)):
            if not json_file.endswith(".json"):
                continue
            
            with open(os.path.join(parsed_dir, json_file)) as f:
                gestures = json.load(f)
            
            for gesture in gestures:
                if gesture.get("type") == "tap":
                    img_path = os.path.join(img_dir, f"{gesture['timestamp']}.png")
                    if os.path.exists(img_path):
                        coords = torch.tensor(gesture["points"][0], dtype=torch.float32)
                        self.samples.append((img_path, coords))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, coords = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, coords


# ============================================================================
# VISION MODELS
# ============================================================================

def create_vision_model(num_classes=2):
    """Create a pre-trained EfficientNet-B0 with custom classifier"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Sigmoid()
    )
    return model


@ray.remote
class VisionModelWorker:
    """Ray remote worker for vision model inference"""
    
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.memory_manager = MemoryManager()
        self.model = self._setup_model()
    
    def _setup_model(self):
        """Initialize the vision model"""
        model = create_vision_model()
        model = model.to(self.device)
        model.eval()
        return model
    
    def forward(self, x):
        """Forward pass with memory management"""
        with self.memory_manager:
            with torch.no_grad():
                x = x.to(self.device)
                output = self.model(x)
                return output.cpu()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model


# ============================================================================
# REWARD SYSTEM
# ============================================================================

class RewardSystem:
    """Unified reward system with multiple reward components"""
    
    def __init__(self, ocr_config: Optional[Dict] = None):
        self.ocr_config = ocr_config or {}
        self.memory_manager = MemoryManager()
        self._ocr_cache = {}
        self._cache_size = 100
    
    def image_difference_reward(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate reward based on image difference"""
        diff = np.abs(img1 - img2).mean() / 255.0
        return -diff
    
    def ocr_score_reward(self, img: np.ndarray, prev_score: Optional[int] = None) -> float:
        """Calculate reward based on OCR score changes"""
        with self.memory_manager:
            try:
                # Check cache
                img_hash = hash(img.tobytes())
                if img_hash in self._ocr_cache:
                    score = self._ocr_cache[img_hash]
                else:
                    text = pytesseract.image_to_string(img)
                    score = self._parse_score_from_text(text)
                    
                    # Cache management
                    if len(self._ocr_cache) >= self._cache_size:
                        oldest_key = next(iter(self._ocr_cache))
                        del self._ocr_cache[oldest_key]
                    self._ocr_cache[img_hash] = score
                
                if score is None or prev_score is None:
                    return 0.0
                
                return (score - prev_score) / 100.0
                
            except Exception as e:
                print(f"OCR error: {e}")
                return 0.0
    
    def _parse_score_from_text(self, text: str) -> Optional[int]:
        """Extract score from OCR text"""
        try:
            numbers = re.findall(r'\d+', text)
            if not numbers:
                return None
            return int(max(numbers, key=len))
        except Exception:
            return None
    
    def multi_objective_reward(self, img1: np.ndarray, img2: np.ndarray, 
                              prev_score: Optional[int] = None) -> float:
        """Calculate combined reward from multiple objectives"""
        image_reward = self.image_difference_reward(img1, img2)
        ocr_reward = self.ocr_score_reward(img2, prev_score) if self.ocr_config.get('use_ocr', True) else 0.0
        return 0.7 * image_reward + 0.3 * ocr_reward


# ============================================================================
# INTRINSIC CURIOSITY MODULE
# ============================================================================

class ICMForwardModel(nn.Module):
    """Forward model for Intrinsic Curiosity Module"""
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, state_feat, action):
        x = torch.cat([state_feat, action], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ICMInverseModel(nn.Module):
    """Inverse model for Intrinsic Curiosity Module"""
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state_feat, next_state_feat):
        x = torch.cat([state_feat, next_state_feat], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class IntrinsicCuriosityModule(nn.Module):
    """Complete Intrinsic Curiosity Module implementation"""
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256, beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.memory_manager = MemoryManager()
        
        self.forward_model = ICMForwardModel(feature_dim, action_dim, hidden_dim)
        self.inverse_model = ICMInverseModel(feature_dim, action_dim, hidden_dim)
    
    def forward(self, state_feat, next_state_feat, action):
        """Forward pass through both models"""
        pred_action = self.inverse_model(state_feat, next_state_feat)
        pred_next_feat = self.forward_model(state_feat, action)
        return pred_action, pred_next_feat
    
    def compute_intrinsic_reward(self, state_feat, next_state_feat, action):
        """Compute intrinsic reward based on prediction error"""
        with torch.no_grad():
            pred_next_feat = self.forward_model(state_feat, action)
            intrinsic_reward = F.mse_loss(pred_next_feat, next_state_feat, reduction='none').mean(dim=1)
        return intrinsic_reward.detach().cpu().numpy()
    
    def compute_loss(self, state_feat, next_state_feat, action):
        """Compute combined ICM loss"""
        pred_action, pred_next_feat = self.forward(state_feat, next_state_feat, action)
        inverse_loss = F.mse_loss(pred_action, action)
        forward_loss = F.mse_loss(pred_next_feat, next_state_feat)
        return (1 - self.beta) * inverse_loss + self.beta * forward_loss


# ============================================================================
# MEMORY SYSTEMS
# ============================================================================

@ray.remote
class FAISSMemoryWorker:
    """Ray remote worker for FAISS-based memory system"""
    
    def __init__(self, embedding_dim: int, max_size: int = 10000, 
                 index_path: Optional[str] = None, use_cosine: bool = False):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.index_path = index_path
        self.use_cosine = use_cosine
        self.memory_manager = MemoryManager()
        
        self.meta = []
        self.index = self._setup_index()
        
        if index_path and os.path.exists(index_path + ".faiss"):
            self.load(index_path)
    
    def _setup_index(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            return None
        
        if self.use_cosine:
            return faiss.IndexFlatIP(self.embedding_dim)
        else:
            return faiss.IndexFlatL2(self.embedding_dim)
    
    def add(self, embeddings: np.ndarray, metadata: List[Any]):
        """Add embeddings and metadata to memory"""
        with self.memory_manager:
            if len(self.meta) + len(embeddings) > self.max_size:
                n_trim = len(self.meta) + len(embeddings) - self.max_size
                self.meta = self.meta[n_trim:]
                self.index.remove_ids(np.arange(n_trim))
            
            self.meta.extend(metadata)
            self.index.add(embeddings)
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Any]]:
        """Search for top-k nearest embeddings"""
        with self.memory_manager:
            distances, indices = self.index.search(query, k)
            return distances, [self.meta[i] for i in indices[0]]
    
    def save(self, path: str):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.meta, f)
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            self.meta = pickle.load(f)
    
    def __len__(self):
        return len(self.meta)


class ReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory_manager = MemoryManager()
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        self.Transition = namedtuple('Transition',
            ('state', 'action', 'reward', 'next_state', 'done'))
    
    def push(self, state, action, reward, next_state, done):
        """Add new experience to buffer"""
        with self.memory_manager:
            max_prio = max(self.priorities) if self.buffer else 1.0
            experience = self.Transition(state, action, reward, next_state, done)
            
            self.buffer.append(experience)
            self.priorities.append(max_prio)
    
    def sample(self, batch_size: int = 32, beta: float = 0.4):
        """Sample batch with prioritized sampling"""
        with self.memory_manager:
            if len(self.buffer) < batch_size:
                return [], None, None
            
            if len(self.buffer) == self.capacity:
                # Prioritized sampling
                prios = np.array(self.priorities)
                probs = prios ** self.alpha
                probs /= probs.sum()
                
                indices = np.random.choice(len(self.buffer), batch_size, p=probs)
                experiences = [self.buffer[idx] for idx in indices]
                
                weights = (len(self.buffer) * probs[indices]) ** (-beta)
                weights /= weights.max()
                
                return experiences, indices, weights
            else:
                # Random sampling
                return random.sample(self.buffer, batch_size), None, None
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        if indices is not None:
            for idx, prio in zip(indices, priorities):
                self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PLANNING SYSTEM
# ============================================================================

class ShortHorizonPlanner:
    """Short-term planning system"""
    
    def __init__(self, max_lookahead: int = 5):
        self.max_lookahead = max_lookahead
        self.memory_manager = MemoryManager()
        
        self.state_buffer = deque(maxlen=max_lookahead)
        self.action_buffer = deque(maxlen=max_lookahead)
        self.reward_buffer = deque(maxlen=max_lookahead)
        
        self._action_cache = {}
        self._cache_size = 100
    
    def plan(self, state: np.ndarray, action: np.ndarray, reward: float) -> np.ndarray:
        """Plan next action based on current state"""
        with self.memory_manager:
            # Check cache
            state_hash = hash(state.tobytes())
            if state_hash in self._action_cache:
                return self._action_cache[state_hash]
            
            # Add to buffers
            self.state_buffer.append(state)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)
            
            # Plan next action
            next_action = self._plan_next_action()
            
            # Cache management
            if len(self._action_cache) >= self._cache_size:
                oldest_key = next(iter(self._action_cache))
                del self._action_cache[oldest_key]
            self._action_cache[state_hash] = next_action
            
            return next_action
    
    def _plan_next_action(self) -> np.ndarray:
        """Implement planning logic (override in subclasses)"""
        # Simple example: return random action
        return np.random.rand(2)


# ============================================================================
# WEB AUTOMATION
# ============================================================================

class WebSearchAgent:
    """Memory-efficient web search agent using Playwright"""
    
    def __init__(self, max_results: int = 5, timeout: int = 30):
        self.bing_url = "https://www.bing.com/search?q="
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.max_results = max_results
        self.timeout = timeout
        self.memory_manager = MemoryManager()
    
    async def search(self, query: str) -> List[Dict[str, str]]:
        """Perform web search and return results"""
        with self.memory_manager:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True, timeout=self.timeout * 1000)
                    page = await browser.new_page()
                    await page.set_extra_http_headers({"User-Agent": self.user_agent})
                    
                    # Navigate and search
                    search_url = self.bing_url + query.replace(' ', '+')
                    await page.goto(search_url, timeout=self.timeout * 1000)
                    await page.wait_for_load_state('networkidle', timeout=self.timeout * 1000)
                    
                    # Extract results
                    results = []
                    elements = await page.query_selector_all('.b_algo')
                    
                    for elem in elements[:self.max_results]:
                        title_elem = await elem.query_selector('h2')
                        if title_elem:
                            link_elem = await title_elem.query_selector('a')
                            if link_elem:
                                url = await link_elem.get_attribute('href')
                                text = await title_elem.text_content()
                                results.append({"title": text, "link": url})
                    
                    await page.close()
                    await browser.close()
                    return results
                    
            except Exception as e:
                print(f"[ERROR] Web search failed: {e}")
                return []


# ============================================================================
# UTILITIES
# ============================================================================

def create_data_loader(batch_size: int = 32, num_workers: int = 0, cache_size: int = 100):
    """Create memory-efficient data loader"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MemoryEfficientImageDataset(IMG_DIR, transform=transform, max_cache_size=cache_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )


def detect_ui_change(prev_img_bytes, curr_img_bytes, threshold: float = 0.05):
    """Detect UI changes between two images"""
    if prev_img_bytes is None or curr_img_bytes is None:
        return False, 0.0, None
    
    img1 = np.array(Image.open(BytesIO(prev_img_bytes)).convert("RGB"))
    img2 = np.array(Image.open(BytesIO(curr_img_bytes)).convert("RGB"))
    
    # Simple difference calculation
    diff = np.abs(img1 - img2).mean() / 255.0
    changed = diff > threshold
    
    return changed, diff, None


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_win_rate(log_dir: str) -> Dict[str, float]:
    """Evaluate win rate from experiment logs"""
    csv_files = glob(os.path.join(log_dir, '*.csv'))
    print(f"Found {len(csv_files)} log files.")
    
    if not csv_files:
        return {}
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        
        # Extract metadata from filename
        base = os.path.basename(f)
        parts = base.replace('.csv', '').split('_')
        
        for p in parts:
            if p.startswith('seed'):
                df['seed'] = int(p[4:])
            if p.startswith('persona'):
                df['persona'] = p[7:]
        
        dfs.append(df)
    
    all_df = pd.concat(dfs, ignore_index=True)
    
    # Define success metric
    if 'success' in all_df.columns:
        all_df['success'] = all_df['success'].astype(int)
    else:
        all_df['success'] = (all_df['reward'] > 0).astype(int)
    
    # Compute win rates
    results = {
        'overall': all_df['success'].mean(),
        'by_persona': all_df.groupby('persona')['success'].mean().to_dict(),
        'by_seed': all_df.groupby('seed')['success'].mean().to_dict()
    }
    
    return results


def run_validation_suite(log_dir: str = 'logs/exp_runs', 
                        min_win_rate: float = 0.1) -> List[Tuple[str, str, float]]:
    """Run validation on all log files"""
    failed = []
    
    for fname in os.listdir(log_dir):
        if not fname.endswith('.csv'):
            continue
        
        fpath = os.path.join(log_dir, fname)
        try:
            df = pd.read_csv(fpath)
            
            # Calculate win rate
            if 'success' in df.columns:
                win_rate = df['success'].mean()
            else:
                win_rate = (df['reward'] > 0).mean()
            
            if win_rate < min_win_rate:
                failed.append((fname, 'win_rate', win_rate))
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            failed.append((fname, 'error', 0.0))
    
    return failed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Models module loaded successfully")
    
    # Test memory manager
    memory_manager = MemoryManager()
    print(f"Memory manager initialized: {memory_manager}")
    
    # Test dataset
    if os.path.exists(IMG_DIR):
        dataset = MemoryEfficientImageDataset(IMG_DIR)
        print(f"Dataset loaded with {len(dataset)} images")
    
    # Test vision model
    model = create_vision_model()
    print(f"Vision model created: {model}")