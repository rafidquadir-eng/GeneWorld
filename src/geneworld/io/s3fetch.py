"""Native parallel S3 fetcher for MitoCheck wells (libcurl multi interface via pycurl).

One thread drives a pycurl CurlMulti with up to `streams` concurrent transfers over reused
HTTPS connections; chunk bodies go to in-memory buffers only. A well's 93 level-1 frames are
decoded and yielded as a (93, 512, 672) uint8 array as soon as all of its chunks arrive; wells
in flight or finished-but-not-yet-consumed are bounded by `max_wells` (the rolling in-memory
buffer; ~32 MB per well); a well's slot is freed only after the consumer has taken it.
Failed chunks are retried with exponential backoff (8 attempts; HTTP 404: 3 attempts); a well
whose chunk keeps failing is yielded with its error instead of an array.

Measured with curl's own parallel mode: 85–91 MB/s from the BIA bucket at 128 streams
(`reports/DATA_ACCESS.md`). Only the BIA bucket holds the whole screen.
"""
from __future__ import annotations

import io
import queue
import random
import threading
import time
from collections.abc import Iterable, Iterator

import numcodecs
import numpy as np
import pycurl

from geneworld.io.mitocheck import LEVEL_SHAPE, N_FRAMES, chunk_url

_BLOSC = numcodecs.Blosc()
MAX_ATTEMPTS = 8
#: completion plan v2 S3 (pre-registered 2026-10-23 for test and screenB): a frame answering
#: HTTP 404 is re-requested with back-off and counts as missing only after 3 attempts (the
#: stage-2 fetch treated 404 as final after 1; S2 re-requested those frames 3 times)
MAX_404_ATTEMPTS = 3


class _Job:
    __slots__ = ("key", "uuid", "well", "bufs", "left", "error", "_done")

    def __init__(self, key, uuid, well):
        self.key, self.uuid, self.well = key, uuid, well
        self.bufs: list[bytes | None] = [None] * N_FRAMES
        self.left = N_FRAMES
        self.error: str | None = None
        self._done = False


def fetch_wells(wells: Iterable[tuple[object, str, str]], level: int = 1, streams: int = 128,
                max_wells: int = 16) -> Iterator[tuple[object, np.ndarray | None, str | None]]:
    """`wells`: iterable of (key, bia_uuid, well e.g. 'A1'). Yields (key, frames, error)."""
    h, w = LEVEL_SHAPE[level]
    out: queue.Queue = queue.Queue()
    slots = threading.BoundedSemaphore(max_wells)
    src = iter(wells)

    def run():
        m = pycurl.CurlMulti()
        free = []
        for _ in range(streams):
            c = pycurl.Curl()
            c.setopt(pycurl.FOLLOWLOCATION, 1)
            c.setopt(pycurl.CONNECTTIMEOUT, 20)
            c.setopt(pycurl.TIMEOUT, 120)
            c.setopt(pycurl.NOSIGNAL, 1)
            free.append(c)
        pending: list[tuple[float, _Job, int, int]] = []    # (not_before, job, t, attempt)
        meta: dict[int, tuple] = {}
        active = 0
        exhausted = False
        jobs_open = 0
        while True:
            # admit new wells while the buffer has room
            while not exhausted and slots.acquire(blocking=False):
                try:
                    key, uuid, well = next(src)
                except StopIteration:
                    slots.release()
                    exhausted = True
                    break
                j = _Job(key, uuid, well)
                jobs_open += 1
                pending.extend((0.0, j, t, 0) for t in range(N_FRAMES))
            now = time.time()
            ready = [p for p in pending if p[0] <= now]
            pending = [p for p in pending if p[0] > now]
            while free and ready:
                _, j, t, att = ready.pop()
                if j.error is not None:
                    continue
                c = free.pop()
                buf = io.BytesIO()
                c.setopt(pycurl.URL, chunk_url(j.uuid, j.well, level, t))
                c.setopt(pycurl.WRITEDATA, buf)
                meta[id(c)] = (j, t, att, buf)
                m.add_handle(c)
                active += 1
            pending.extend(ready)
            if active:
                m.select(0.5)
                while True:
                    ret, _ = m.perform()
                    if ret != pycurl.E_CALL_MULTI_PERFORM:
                        break
                _, ok, bad = m.info_read()
                for c in ok:
                    j, t, att, buf = meta.pop(id(c))
                    code = c.getinfo(pycurl.RESPONSE_CODE)
                    m.remove_handle(c)
                    free.append(c)
                    active -= 1
                    if code == 200:
                        j.bufs[t] = buf.getvalue()
                        j.left -= 1
                    else:
                        _retry(j, t, att, f"HTTP {code}", pending,
                               limit=MAX_404_ATTEMPTS if code == 404 else MAX_ATTEMPTS)
                    jobs_open -= _maybe_finish(j, out, slots, h, w)
                for c, errno, msg in bad:
                    j, t, att, _ = meta.pop(id(c))
                    m.remove_handle(c)
                    free.append(c)
                    active -= 1
                    _retry(j, t, att, f"curl {errno}: {msg}", pending)
                    jobs_open -= _maybe_finish(j, out, slots, h, w)
            elif exhausted and jobs_open == 0 and not pending:
                break
            else:
                time.sleep(0.05)
        for c in free:
            c.close()
        m.close()
        out.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (item := out.get()) is not None:
        yield item
        slots.release()      # the consumer has taken the well: it leaves the buffer


def _retry(j, t, att, err, pending, limit=MAX_ATTEMPTS):
    if att + 1 >= limit:
        j.error = f"frame {t}: {err} after {att + 1} attempts"
        return
    delay = min(30.0, 0.5 * 2 ** att) * (0.5 + random.random())
    pending.append((time.time() + delay, j, t, att + 1))


def _maybe_finish(j, out, slots, h, w) -> int:
    """Emit a finished (or failed) well once; returns 1 when the well leaves the buffer."""
    if j._done:
        return 0
    if j.error is not None:
        j._done = True
        out.put((j.key, None, j.error))
        return 1
    if j.left == 0:
        j._done = True
        frames = np.empty((N_FRAMES, h, w), np.uint8)
        for i, b in enumerate(j.bufs):
            frames[i] = np.frombuffer(_BLOSC.decode(b), np.uint8).reshape(h, w)
        j.bufs = []
        out.put((j.key, frames, None))
        return 1
    return 0
