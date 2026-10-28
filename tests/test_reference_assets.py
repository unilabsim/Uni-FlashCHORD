from __future__ import annotations

import numpy as np
import pytest

from chord_unilab.data.reference_assets import (
    load_reference_manifest,
    reference_coacd_cache,
)


def test_reference_manifest_contains_tissue_demo_closure() -> None:
    manifest = load_reference_manifest()
    assert {entry.sequence_id for entry in manifest.sequences} == {"tissue_box_simple"}
    for entry in manifest.sequences:
        assert entry.parquet_path.is_file()
        assert entry.support_path is not None and entry.support_path.is_file()
        assert entry.object_asset_paths
        assert all(path.is_file() for path in entry.object_asset_paths)

    tissue = manifest.sequence("tissue_box_simple")
    assert tissue.coacd_cache is not None
    assert tissue.coacd_cache.content_key == "2daf6f9482cb02d2"
    with np.load(tissue.coacd_cache.path) as cache:
        assert int(cache["n"]) == tissue.coacd_cache.hull_count == 54


def test_reference_coacd_cache_requires_matching_content_key() -> None:
    manifest = load_reference_manifest()
    cache = reference_coacd_cache(
        "tissue_box_simple", "2daf6f9482cb02d2", manifest=manifest
    )
    assert cache is not None and cache.hull_count == 54
    with pytest.raises(ValueError, match="CoACD key mismatch"):
        reference_coacd_cache("tissue_box_simple", "wrong", manifest=manifest)
