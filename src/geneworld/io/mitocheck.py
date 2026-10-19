"""MitoCheck (IDR idr0013 / BioImage Archive S-BIAD865) access by well, streamed from EBI S3.

The BioImage Archive keeps every MitoCheck plate as an unzipped OME-Zarr store on EBI's S3
endpoint (bucket ``bia-integrator-data``, prefix ``S-BIAD865/<uuid>/<uuid>.zarr``). A well is
``<row>/<col>/0`` with pyramid levels 0–3 (1024x1344 down to 128x168), axes (t, c, z, y, x),
93 frames at 30 min, one chunk per frame, Blosc-LZ4 uint8. ``read_well`` fetches one level of
one well with parallel GETs and returns a (T, H, W) uint8 array; nothing is written to disk.

Measured on 2026-10-19 (`reports/DATA_ACCESS.md`): EBI S3 37.8 MB/s at 128 streams with no
errors; EBI HTTP byte ranges into the per-plate zips fail above ~16 connections.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import requests

S3 = "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD865"
REPO = Path(__file__).resolve().parents[3]
PLATE_MAP = REPO / "data" / "meta" / "s3_plate_map.json"
N_FRAMES = 93
LEVEL_SHAPE = {0: (1024, 1344), 1: (512, 672), 2: (256, 336), 3: (128, 168)}

_BLOSC = numcodecs.Blosc()
_SESSION: requests.Session | None = None


def session(pool: int = 256) -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=pool,
                                                          max_retries=5))
        _SESSION = s
    return _SESSION


def plate_map() -> dict:
    return json.loads(PLATE_MAP.read_text())


def well_path(well: str) -> str:
    """'A1' -> 'A/1' (the OME-Zarr row/column layout)."""
    return f"{well[0]}/{int(well[1:])}"


def chunk_url(uuid: str, well: str, level: int, t: int) -> str:
    return f"{S3}/{uuid}/{uuid}.zarr/{well_path(well)}/0/{level}/{t}/0/0/0/0"


def _get(url: str) -> bytes:
    import random
    import time
    last = None
    for attempt in range(8):
        try:
            r = session().get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            last = r.status_code
            if r.status_code == 404:
                break
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(min(20.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    raise OSError(f"GET {url} failed ({last})")


def read_well(uuid: str, well: str, level: int = 1, frames: range | None = None,
              pool: ThreadPoolExecutor | None = None) -> np.ndarray:
    """(T, H, W) uint8 for one well at one pyramid level."""
    frames = frames or range(N_FRAMES)
    h, w = LEVEL_SHAPE[level]
    urls = [chunk_url(uuid, well, level, t) for t in frames]
    if pool is None:
        with ThreadPoolExecutor(32) as ex:
            blobs = list(ex.map(_get, urls))
    else:
        blobs = list(pool.map(_get, urls))
    out = np.empty((len(urls), h, w), np.uint8)
    for i, b in enumerate(blobs):
        out[i] = np.frombuffer(_BLOSC.decode(b), np.uint8).reshape(h, w)
    return out
