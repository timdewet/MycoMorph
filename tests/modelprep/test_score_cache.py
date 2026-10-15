"""Per-(model, HDF5) score cache (review M5)."""

from __future__ import annotations

import os

import numpy as np

from modelprep.core import score_cache
from modelprep.core.score_cache import ScoreCache


def _files(tmp_path):
    h5 = tmp_path / "all_crops.h5"
    model = tmp_path / "model.pth"
    h5.write_bytes(b"h5")
    model.write_bytes(b"weights-v1")
    return h5, model


def test_cache_round_trips_through_disk(tmp_path):
    h5, model = _files(tmp_path)
    root = tmp_path / "cache"
    c = ScoreCache.open(h5, model, 5, root=root)
    assert c.missing(np.arange(5)).tolist() == [0, 1, 2, 3, 4]
    c.scores[[1, 3]] = [0.25, 0.75]
    c.save()

    score_cache.clear_memory()          # force a disk read
    c2 = ScoreCache.open(h5, model, 5, root=root)
    assert c2.missing(np.arange(5)).tolist() == [0, 2, 4]
    assert c2.scores[3] == np.float32(0.75)


def test_retrained_model_gets_a_fresh_entry(tmp_path):
    h5, model = _files(tmp_path)
    root = tmp_path / "cache"
    c = ScoreCache.open(h5, model, 3, root=root)
    c.scores[:] = 0.5
    c.save()
    model.write_bytes(b"weights-v2-longer")      # new size → new key
    os.utime(model, ns=(1, 1))
    c2 = ScoreCache.open(h5, model, 3, root=root)
    assert c2.key != c.key
    assert c2.missing(np.arange(3)).size == 3
