"""Read members of a remote zip archive by HTTP byte ranges (no full download).

The per-plate MitoCheck OME-Zarr archives on the BioImage Archive (S-BIAD865) are 26–31 GB zip
files. ``RemoteFile`` gives ``zipfile.ZipFile`` a seekable file object backed by HTTP Range
requests, so the central directory and individual chunk files (members) can be read without
fetching the archive. Members of OME-Zarr zips are normally stored uncompressed (method 0), so
a member's bytes are one contiguous range: ``member_range`` returns it, and
``fetch_ranges`` downloads many such ranges with a pool of parallel connections.
"""
from __future__ import annotations

import io
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

_SESSION = None


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=512,
                                          max_retries=5)
        s.mount("https://", a)
        s.mount("http://", a)
        _SESSION = s
    return _SESSION


def get_range(url: str, start: int, end: int, timeout: float = 120.0) -> bytes:
    """Bytes [start, end] inclusive."""
    import random
    import time
    last = None
    for attempt in range(10):
        try:
            r = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout)
            if r.status_code in (200, 206) and len(r.content) == end - start + 1:
                return r.content
            last = r.status_code
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(min(30.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
    raise OSError(f"range {start}-{end} of {url} failed ({last})")


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges, with a small read-ahead cache."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        h = session().head(url, timeout=60, allow_redirects=True)
        self.size = int(h.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            s = i * self.block
            self._cache[i] = get_range(self.url, s, min(self.size, s + self.block) - 1)
        return self._cache[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            b = self._block(i)[o:o + n]
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def open_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(RemoteFile(url))


def member_range(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    """Absolute byte range [start, end] of a stored member's data."""
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{info.filename} is compressed (method {info.compress_type})")
    fp = zf.fp
    fp.seek(info.header_offset)
    hdr = fp.read(30)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    return start, start + info.file_size - 1


def fetch_ranges(url: str, ranges: list[tuple[int, int]], workers: int = 128) -> list[bytes]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda r: get_range(url, r[0], r[1]), ranges))
