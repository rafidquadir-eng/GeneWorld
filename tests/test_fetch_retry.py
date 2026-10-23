"""S3 rule in the fetcher: HTTP 404 frames count as missing only after 3 attempts."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.io import s3fetch  # noqa: E402


class _SyntheticJob:
    error = None


def attempts_until_error(limit: int) -> int:
    j, pending, att = _SyntheticJob(), [], 0
    while True:
        s3fetch._retry(j, 0, att, "HTTP 404", pending, limit=limit)
        if j.error:
            return att + 1
        att = pending.pop()[3]


def test_404_gets_three_attempts():
    assert attempts_until_error(s3fetch.MAX_404_ATTEMPTS) == 3
    assert "after 3 attempts" in _job_error(s3fetch.MAX_404_ATTEMPTS)


def test_other_errors_keep_eight_attempts():
    assert attempts_until_error(s3fetch.MAX_ATTEMPTS) == 8


def _job_error(limit: int) -> str:
    j, pending = _SyntheticJob(), []
    s3fetch._retry(j, 0, limit - 1, "HTTP 404", pending, limit=limit)
    return j.error
