# data_refactored.py
"""
Refactored data collection and analysis utilities with improved memory management,
lazy loading, and better organization.
"""
import os
import io
import json
import gc
import time
import asyncio
from datetime import datetime
from typing import Dict, Tuple, Optional, AsyncIterator, List

import psutil
import adbutils
import numba
import aiofiles
from lxml import etree
from PIL import Image

import re
_UP_RE = re.compile(r"ABS_MT_TRACKING_ID\s+ffffffff")
_X_RE = re.compile(r"ABS_MT_POSITION_X\s+([0-9a-f]+)")
_Y_RE = re.compile(r"ABS_MT_POSITION_Y\s+([0-9a-f]+)")
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

DATA_DIR = "collected_data"
OUTPUT_DIR = "parsed_data"
CONFIG_FILE = "device_config.json"
SCREENSHOT_DIR = os.path.join(DATA_DIR, "screenshots")
XML_DIR = os.path.join(DATA_DIR, "xml")
EVENT_DIR = os.path.join(DATA_DIR, "events")

_pd = None
_plt = None
_sns = None

def _get_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd

def _get_matplotlib():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

def _get_seaborn():
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns

def cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def memory_usage_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2

def memory_limit_exceeded(limit_mb: float = 1024) -> bool:
    return memory_usage_mb() > limit_mb

class ConfigManager:
    _instance = None
    _config: Optional[Dict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, path: str = CONFIG_FILE) -> Dict:
        if self._config is None:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except FileNotFoundError:
                self._config = self._default_config()
                self.save(path)
        return self._config

    def save(self, path: str = CONFIG_FILE):
        if self._config is not None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)

    def _default_config(self) -> Dict:
        try:
            device = adbutils.device()
            w, h = device.window_size()
        except Exception:
            w, h = 1080, 1920
        return {"screen_width": w, "screen_height": h, "action_cooldown": 0.1, "memory_limit_mb": 1024}

    def screen_size(self) -> Tuple[int, int]:
        cfg = self.load()
        return cfg.get("screen_width", 1080), cfg.get("screen_height", 1920)

    def set_screen_size(self, width: int, height: int):
        self._config = self.load()
        self._config.update({"screen_width": width, "screen_height": height})
        self.save()

@numba.jit(nopython=True)
def normalize_point(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    return min(1.0, max(0.0, x/w)), min(1.0, max(0.0, y/h))

@numba.jit(nopython=True)
def denormalize_point(xn: float, yn: float, w: int, h: int) -> Tuple[int, int]:
    return int(xn*w), int(yn*h)

@numba.jit(nopython=True)
def point_in_bounds(x: int, y: int, bounds: Tuple[int,int,int,int]) -> bool:
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2

def init_dirs():
    for p in [DATA_DIR, OUTPUT_DIR, SCREENSHOT_DIR, XML_DIR, EVENT_DIR]:
        os.makedirs(p, exist_ok=True)

class ADBDeviceManager:
    def __init__(self, max_attempts: int = 3):
        self._device = None
        self._attempts = 0
        self._max = max_attempts

    @property
    def device(self):
        while self._device is None and self._attempts < self._max:
            try:
                self._device = adbutils.device()
            except Exception as e:
                self._attempts += 1
                time.sleep(1)
                if self._attempts >= self._max:
                    raise ConnectionError(f"ADB connect failed: {e}")
        self._attempts = 0
        return self._device

    def reset(self):
        self._device = None
        self._attempts = 0

def _install_uvloop():
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass

_install_uvloop()

class AsyncActionExecutor:
    def __init__(self):
        self.config = ConfigManager()
        self.adb = ADBDeviceManager()
        self._last = time.time()

    async def _rate_limit(self):
        cfg = self.config.load()
        cooldown = cfg.get("action_cooldown", 0.1)
        now = time.time()
        wait = cooldown - (now - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.time()

    async def tap(self, x: int, y: int):
        await self._rate_limit()
        try:
            self.adb.device.click(x, y)
        finally:
            cleanup_memory()

    async def tap_norm(self, xn: float, yn: float):
        w, h = self.config.screen_size()
        x, y = denormalize_point(xn, yn, w, h)
        await self.tap(x, y)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, dur: int = 500):
        await self._rate_limit()
        try:
            self.adb.device.swipe(x1, y1, x2, y2, duration=dur/1000)
        finally:
            cleanup_memory()

    async def swipe_norm(self, x1n: float, y1n: float, x2n: float, y2n: float, dur: int = 500):
        w, h = self.config.screen_size()
        x1, y1 = denormalize_point(x1n, y1n, w, h)
        x2, y2 = denormalize_point(x2n, y2n, w, h)
        await self.swipe(x1, y1, x2, y2, dur)

    async def screenshot(self) -> Optional[bytes]:
        try:
            pil_img = self.adb.device.screenshot()
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG', optimize=True)
            return buf.getvalue()
        except Exception:
            self.adb.reset()
            return None
        finally:
            cleanup_memory()

    async def dump_ui_xml(self) -> str:
        try:
            return self.adb.device.dump_hierarchy() or ''
        except Exception:
            self.adb.reset()
            return ''
        finally:
            cleanup_memory()

class LazyEventParser:
    def __init__(self, path: str, batch: int = 1000):
        self.path = path
        self.batch = batch
        self._x = None
        self._y = None

    async def parse(self) -> AsyncIterator[Dict]:
        w, h = ConfigManager().screen_size()
        async with aiofiles.open(self.path, 'r', encoding='utf-8') as f:
            count = 0
            async for line in f:
                count += 1
                xm = _X_RE.search(line)
                ym = _Y_RE.search(line)
                if xm:
                    self._x = int(xm.group(1), 16)
                if ym:
                    self._y = int(ym.group(1), 16)
                if _UP_RE.search(line) and self._x is not None and self._y is not None:
                    yield {
                        "type": "tap",
                        "timestamp": os.path.basename(self.path),
                        "abs_point": [self._x, self._y],
                        "norm_point": normalize_point(self._x, self._y, w, h)
                    }
                    self._x = None
                    self._y = None
                if count % self.batch == 0 and memory_limit_exceeded():
                    cleanup_memory()
                    await asyncio.sleep(0.1)

class XMLParser:
    def __init__(self, path: str):
        self.path = path

    async def parse(self) -> AsyncIterator[Dict]:
        try:
            for event, elem in etree.iterparse(self.path, events=('end',)):
                if elem.tag == 'node':
                    bounds = elem.get('bounds')
                    bb = _BOUNDS_RE.match(bounds).groups() if bounds else None
                    if bb:
                        coords = tuple(map(int, bb))
                        yield {
                            "resource_id": elem.get('resource-id', ''),
                            "class": elem.get('class', ''),
                            "text": elem.get('text', ''),
                            "clickable": elem.get('clickable') == 'true',
                            "bounds": coords
                        }
                parent = elem.getparent()
                elem.clear()
                if parent is not None:
                    while parent.getprevious() is not None:
                        del parent.getparent()[0]
                if memory_limit_exceeded():
                    cleanup_memory()
                    await asyncio.sleep(0.01)
        except Exception:
            pass

def find_element(gesture: Dict, elements: List[Dict]) -> Tuple[Optional[Dict], str]:
    if 'abs_point' in gesture:
        x, y = gesture['abs_point']
    else:
        x, y = denormalize_point(*gesture['norm_point'], *ConfigManager().screen_size())
    inside = []
    closest = None
    min_dist = float('inf')
    for el in elements:
        bounds = el['bounds']
        if point_in_bounds(x, y, bounds):
            area = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
            inside.append((el, area))
        else:
            cx = (bounds[0]+bounds[2])//2
            cy = (bounds[1]+bounds[3])//2
            dist = (cx-x)**2+(cy-y)**2
            if dist < min_dist:
                min_dist = dist
                closest = el
    if inside:
        el, _ = min(inside, key=lambda i: i[1])
        return el, 'inside'
    return (closest, 'closest') if closest else (None, '')

class DataCollector:
    def __init__(self):
        init_dirs()
        self.exec = AsyncActionExecutor()

    async def run(self):
        proc = await asyncio.create_subprocess_exec(
            'adb','shell','getevent','-l',
            stdout=asyncio.subprocess.PIPE
        )
        buf = []
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='ignore')
                buf.append(line)
                if _UP_RE.search(line):
                    png, xml = await asyncio.gather(
                        self.exec.screenshot(),
                        self.exec.dump_ui_xml()
                    )
                    if png:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        await asyncio.gather(
                            self._write(SCREENSHOT_DIR, f'{ts}.png', png),
                            self._write(XML_DIR, f'{ts}.xml', xml),
                            self._write(EVENT_DIR, f'{ts}.log', ''.join(buf))
                        )
                        buf.clear()
                        cleanup_memory()
        except KeyboardInterrupt:
            proc.kill()
        finally:
            cleanup_memory()

    async def _write(self, dir_path: str, filename: str, data):
        path = os.path.join(dir_path, filename)
        mode = 'wb' if isinstance(data, bytes) else 'w'
        async with aiofiles.open(
            path, mode,
            encoding=None if isinstance(data, bytes) else 'utf-8'
        ) as f:
            await f.write(data)

class LogAnalyzer:
    def __init__(self, path: str, out_dir: str = 'analysis'):
        self.path = path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    async def analyze(self):
        pd = _get_pandas()
        plt = _get_matplotlib()
        sns = _get_seaborn()
        df = pd.read_csv(self.path)
        if 'step' in df and 'reward' in df:
            plt.figure()
            plt.plot(df['step'], df['reward'])
            plt.savefig(os.path.join(self.out_dir, 'reward_curve.png'))
            plt.close()
        cleanup_memory()

def run_data_collector():
    asyncio.run(DataCollector().run())

def analyze_logs(path: str):
    asyncio.run(LogAnalyzer(path).analyze())

def get_screen_size():
    return ConfigManager().screen_size()

def save_config(w: int, h: int):
    ConfigManager().set_screen_size(w, h)
