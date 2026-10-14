import base64
import html
import json
import re

import numpy as np
import pytest
from PIL import Image

from roll.pipeline.agentic.env.city_grounding.verify_cache import _project
from scripts.demo import view_3d


GROUND, SIZE, REGION, COLOUR = 5.0, 64, 40.0, (120, 160, 200)


def _camera(target, azimuth, elevation=30.0, factor=1.3):
    """Same pinhole convention as render_open3d._make_pinhole_params, without Open3D."""
    az, el = np.radians([azimuth, elevation])
    distance = factor * REGION
    eye = target + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    forward = (target - eye) / distance
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = [right, np.cross(forward, right), forward]
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
    focal = SIZE * distance / REGION
    return extrinsic, np.array([[focal, 0, SIZE / 2 - 0.5], [0, focal, SIZE / 2 - 0.5], [0, 0, 1.0]]), eye


def _ground_depth(extrinsic, intrinsic, eye):
    """Camera depth of the plane z = GROUND at every pixel."""
    u, v = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    rays = np.stack([(u - intrinsic[0, 2]) / intrinsic[0, 0], (v - intrinsic[1, 2]) / intrinsic[1, 1],
                     np.ones_like(u, dtype=float)], axis=-1) @ extrinsic[:3, :3]
    return (GROUND - eye[2]) / rays[..., 2]


@pytest.fixture(params=["png", "npz"])
def cache(tmp_path, request):
    """A flat, single-colour scene cached as 3 x 3 regions in either lossless depth layout."""
    root = tmp_path / "scene"
    for name in ("imgs", "cams", "depths"):
        (root / name).mkdir(parents=True)
    centres = [(x, y) for x in (-15.0, 0.0, 15.0) for y in (-15.0, 0.0, 15.0)]
    np.save(root / "regions.npy", np.array([[x - 20, y - 20, x + 20, y + 20] for x, y in centres], np.float32))
    for entry, (x, y) in enumerate(centres):
        cameras = [_camera(np.array([x, y, GROUND + 3]), azimuth) for azimuth in (0, 90, 180, 270)]
        depths = np.stack([np.round(_ground_depth(*camera) / 0.05) for camera in cameras]).astype(np.uint16)
        np.savez(root / "cams" / f"{entry:06d}.npz", extrinsics=np.stack([camera[0] for camera in cameras]),
                 intrinsics=np.stack([camera[1] for camera in cameras]))
        if request.param == "npz":
            np.savez(root / "depths" / f"{entry:06d}.npz", depths=depths)
        for depth, cardinal in zip(depths, view_3d.CARDINALS):
            Image.new("RGB", (SIZE, SIZE), COLOUR).save(root / "imgs" / f"{entry:06d}_{cardinal}.jpg")
            if request.param == "png":
                Image.fromarray(depth).save(root / "depths" / f"{entry:06d}_{cardinal}.png")
    (root / "scene_meta.json").write_text(json.dumps({
        "scene_id": "scene", "region_size": REGION, "step": 15.0, "output_size": SIZE,
        "save_depth": True, "depth_scale": 0.05, "depth_format": request.param}))
    return tmp_path


def test_back_projection_inverts_the_cache_projection():
    extrinsic, intrinsic, _ = _camera(np.array([3.0, -2.0, 8.0]), 90)
    depth = np.random.default_rng(0).uniform(20, 80, (SIZE, SIZE)).astype(np.float32)
    world = view_3d._back_project(depth, extrinsic, intrinsic, (SIZE, SIZE))
    pixels, camera_depth = _project(world, extrinsic, intrinsic)
    u, v = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    np.testing.assert_allclose(pixels, np.stack([u, v], axis=-1).reshape(-1, 2), atol=1e-3)
    np.testing.assert_allclose(camera_depth, depth.ravel(), rtol=1e-5)


def test_rebuild_scene_recovers_the_cached_ground(cache):
    box = [-4.0, -3.0, 4.0, 3.0]
    scene = view_3d.rebuild_scene(cache, "scene", box, radius=20.0, voxel=0.5)
    assert np.abs(scene.xyz[:, 2] - GROUND).max() < 0.06
    assert np.linalg.norm(scene.xyz[:, :2], axis=1).max() <= 20.0
    assert np.abs(scene.rgb.astype(int) - COLOUR).max() <= 6
    # No 2 m cell of the inner disc is left empty by the region ownership.
    cells = {tuple(cell) for cell in np.floor(scene.xyz[:, :2] / 2).astype(int)}
    assert all((i, j) in cells for i in range(-8, 8) for j in range(-8, 8) if np.hypot(i + 0.5, j + 0.5) * 2 < 17)
    assert scene.zs == pytest.approx((GROUND, GROUND + 0.3), abs=0.06)
    assert 0.3 < scene.spacing < 1.0


def test_viewer_embeds_the_rebuilt_scene(cache):
    scene = view_3d.rebuild_scene(cache, "scene", [-4.0, -3.0, 4.0, 3.0], radius=20.0, voxel=0.5)
    mapping = {"x_min": -60.0, "y_min": -40.0, "x_max": 60.0, "y_max": 40.0}
    page = view_3d.viewer_html(scene, title="Flat", query="a </script> b", bev=(Image.new("RGB", (120, 80)), mapping),
                               zooms=[[-30.0, -20.0, 30.0, 20.0]])
    document = html.unescape(re.search(r'<iframe srcdoc="(.*?)" style=', page, re.S).group(1))
    assert document.count("</script>") == 1
    data = json.loads(re.search(r"const D = (\{.*\});\n", document).group(1))
    assert data["count"] == len(scene.xyz)
    assert data["box"][:4] == [-4.0, -3.0, 4.0, 3.0] and data["zooms"] == [[-30.0, -20.0, 30.0, 20.0]]
    # The box is centred on the local origin, so the quantised points decode back to world metres.
    origin = np.array([0.0, 0.0, scene.zs[0] - data["box"][4]])
    quantised = np.frombuffer(base64.b64decode(data["positions"]), "<u2").reshape(-1, 3)
    np.testing.assert_allclose(origin + data["low"] + quantised / 65535 * data["span"], scene.xyz, atol=2e-3)
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["colours"]), np.uint8).reshape(-1, 3), scene.rgb)
    assert data["bev"]["bounds"] == [-60.0, -40.0, 60.0, 40.0] and data["query"] == "a </script> b"
