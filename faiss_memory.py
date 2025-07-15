import os
import time
import gc
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import psutil
import memory_profiler
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class FAISSMemory:
    def __init__(
        self,
        embedding_dim: int,
        max_size: int = 10000,
        index_path: Optional[str] = None,
        use_cosine: bool = False,
        memory_threshold: float = 0.8,
        cleanup_interval: float = 30.0
    ):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.meta: List[Any] = []
        self.index_path = index_path
        self.use_cosine = use_cosine
        self.memory_threshold = memory_threshold
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval
        self.index = None
        if FAISS_AVAILABLE:
            self._setup_index()
            if index_path and os.path.exists(index_path + ".faiss"):
                self.load(index_path)

    def _setup_index(self) -> None:
        metric = faiss.METRIC_INNER_PRODUCT if self.use_cosine else faiss.METRIC_L2
        self.index = faiss.IndexFlat(self.embedding_dim, metric)

    def _monitor_memory(self) -> None:
        proc = psutil.Process()
        if proc.memory_info().rss / psutil.virtual_memory().total > self.memory_threshold:
            self._cleanup()

    def _should_cleanup(self) -> bool:
        return (
            time.time() - self._last_cleanup > self._cleanup_interval or
            len(self.meta) > self.max_size
        )

    def _cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory_profiler.memory_usage()
        self._last_cleanup = time.time()

    def add(self, embeddings: np.ndarray, metadata: List[Any]) -> None:
        if not FAISS_AVAILABLE:
            return
        if self._should_cleanup():
            self._cleanup()
        total = len(self.meta) + len(metadata)
        if total > self.max_size:
            trim = total - self.max_size
            self.meta = self.meta[trim:]
            if hasattr(self.index, 'remove_ids'):
                self.index.remove_ids(np.arange(trim))
        self.index.add(embeddings)
        self.meta.extend(metadata)
        self._monitor_memory()

    def search(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, List[Any]]:
        if not FAISS_AVAILABLE or self.index is None:
            return np.empty((0,)), []
        if self._should_cleanup():
            self._cleanup()
        distances, idxs = self.index.search(query, k)
        results = []
        for row in idxs:
            results.append([self.meta[i] for i in row if i < len(self.meta)])
        return distances, results

    def save(self, path: str) -> None:
        if not FAISS_AVAILABLE or self.index is None:
            return
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.meta, f)

    def load(self, path: str) -> None:
        if not FAISS_AVAILABLE:
            return
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            self.meta = pickle.load(f)

    def __len__(self) -> int:
        return len(self.meta)

if __name__ == "__main__":
    mem = FAISSMemory(embedding_dim=8, use_cosine=True)
    vecs = np.random.rand(5, 8).astype(np.float32)
    mem.add(vecs, [f"id_{i}" for i in range(5)])
    d, res = mem.search(vecs, k=3)
    print("Search:", res)
    mem.save("test_index")
    mem2 = FAISSMemory(embedding_dim=8, index_path="test_index", use_cosine=True)
    print("Reload:", mem2.search(vecs, k=3))
