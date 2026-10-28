"""Object asset conversion and Newton integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chord_unilab.assets import convert_object


def test_object_fragment_loads_in_unisim_newton(tmp_path: Path) -> None:
    """The rigid tissue fragment materializes and steps through Newton."""
    torch = pytest.importorskip("torch")
    try:
        from unisim.backend.newton.dependencies import (
            NewtonDependencyError,
            load_newton_dependencies,
        )
    except ImportError as exc:  # pragma: no cover - unisim without newton extra
        pytest.skip(f"unisim newton backend unavailable: {exc}")
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("newton smoke requires a CUDA Warp device")

    import unisim
    from unisim.scene import SceneCfg

    fragment = convert_object.convert_object_from_parquet(
        convert_object.DEFAULT_PARQUET, cache_dir=tmp_path
    )
    backend = unisim.create_backend(
        "newton",
        scene=SceneCfg(model_file=str(fragment)),
        num_envs=1,
        sim_dt=0.01,
        newton_device=str(device),
        newton_use_cuda_graph=False,
    )
    try:
        backend.materialize()
        ctrl = torch.zeros(
            (1, backend.num_actuators), dtype=torch.float32, device=str(device)
        )
        for _ in range(5):
            backend.step_tensor(ctrl, nsteps=1)
        layout = backend.get_physics_state_layout()
        parts = layout.split_state(backend.get_physics_state())
        assert parts.qpos.shape == (1, 7)
        assert parts.qvel.shape == (1, 6)
        assert np.isfinite(parts.qpos).all() and np.isfinite(parts.qvel).all()
    finally:
        backend.cleanup_scene_assets()


def test_collision_hulls_preserve_cached_array_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import coacd

    source = tmp_path / "source.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        '<robot name="object"><link name="object">'
        '<visual><geometry><mesh filename="source.obj"/></geometry></visual>'
        '<collision><origin xyz="0.1 0.2 0.3"/>'
        '<geometry><mesh filename="source.obj" scale="0.5 1 2"/></geometry>'
        "</collision></link></robot>"
    )
    vertices = np.array(
        [[0.123456789, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    calls = []

    def fake_coacd(mesh, **params):
        calls.append((mesh, params))
        return [(vertices, faces), (vertices + 1.0, faces)]

    monkeypatch.setattr(coacd, "run_coacd", fake_coacd)
    converted = convert_object._convert_urdf(urdf, tmp_path)
    assert len(calls) == 1
    assert calls[0][1] == convert_object.COACD_PARAMS

    geoms = list(converted.iter("geom"))
    visual = [geom for geom in geoms if geom.get("contype") == "0"]
    collision = [geom for geom in geoms if geom.get("contype") != "0"]
    assert len(visual) == 1
    assert len(collision) == 2
    for geom in collision:
        np.testing.assert_allclose(
            [float(value) for value in str(geom.get("pos")).split()], [0.1, 0.2, 0.3]
        )
    asset = converted.find("asset")
    assert asset is not None
    meshes = {mesh.get("name"): mesh for mesh in asset.findall("mesh")}
    assert str(meshes[visual[0].get("mesh")].get("file")).endswith("source.obj")
    hull_files = []
    for geom in collision:
        mesh = meshes[geom.get("mesh")]
        assert mesh.get("scale") == "0.5 1 2"
        assert "maxhullvert" not in mesh.attrib
        hull_files.append(Path(str(mesh.get("file"))))
    assert [path.name for path in hull_files] == ["hull_000.obj", "hull_001.obj"]

    for index, path in enumerate(hull_files):
        loaded_vertices = []
        loaded_faces = []
        for line in path.read_text().splitlines():
            if line.startswith("v "):
                loaded_vertices.append([float(value) for value in line.split()[1:]])
            elif line.startswith("f "):
                loaded_faces.append([int(value) - 1 for value in line.split()[1:]])
        np.testing.assert_array_equal(
            np.asarray(loaded_vertices, dtype=np.float32), vertices + index
        )
        np.testing.assert_array_equal(np.asarray(loaded_faces, dtype=np.int32), faces)


def test_convex_cache_miss_has_a_single_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import coacd

    source = tmp_path / "source.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    calls = []

    def fake_coacd(mesh, **params):
        calls.append((mesh, params))
        time.sleep(0.05)
        return [(vertices, faces)]

    monkeypatch.setattr(coacd, "run_coacd", fake_coacd)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: convert_object.convex_hulls(source, cache_dir=tmp_path),
                range(4),
            )
        )
    assert len(calls) == 1
    for hulls in results:
        np.testing.assert_array_equal(hulls[0][0], vertices)
        np.testing.assert_array_equal(hulls[0][1], faces)
