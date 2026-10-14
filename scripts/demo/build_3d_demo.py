#!/usr/bin/env python3
"""Render global BEV, zoomed BEV and an oblique view from a real point cloud.

The camera path is editorial. Only XY comes from the recorded prediction; the
display box's Z extent is estimated from raw points, without ground-truth boxes.
Run with the existing release environment; see scripts/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy import ndimage

WORKSPACE = Path(__file__).resolve().parents[3]
WIDTH, HEIGHT, FPS = 1280, 720, 20
ZOOM_START, ZOOM_DURATION, ZOOM_HOLD = 1.25, 2.75, 0.4
OBLIQUE_START = ZOOM_START + ZOOM_DURATION + ZOOM_HOLD
OBLIQUE_DURATION = 2.5
OBLIQUE_END = OBLIQUE_START + OBLIQUE_DURATION
PREDICTION_START = OBLIQUE_END - 0.5
SECONDS = OBLIQUE_END + 1.5
FRAME_COUNT = round(SECONDS * FPS)
POSTER_FRAME = round((OBLIQUE_END + 0.3) * FPS)
GIF_WIDTH, GIF_FPS, GIF_COLORS = 448, 20, 96
GIF_LOSSY = 80
GIF_FILE = "demo-flow.gif"
BACKGROUND = (0.035, 0.047, 0.065, 1.0)
GOLD = (1.0, 0.36, 0.04)
GENTLE = {
    "pan_start_seconds": .4, "pan_complete_seconds": 2.3,
    "zoom_start_seconds": 2.3, "zoom_complete_seconds": 6.9,
    "zoom_hold_seconds": 0., "oblique_start_seconds": 6.9,
    "oblique_complete_seconds": 10.9, "prediction_start_seconds": 6.9,
}
GENTLE_REVISION = 2
GENTLE_AZIMUTH_OFFSET = 15.0
PREDICTION_FADE_SECONDS = .3
LOOP_FADE_SECONDS = .35
BOX_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
             (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
BOX_FACES = [([0,1,3,2],[-1,0,0]), ([4,6,7,5],[1,0,0]),
             ([0,4,5,1],[0,-1,0]), ([2,3,7,6],[0,1,0]),
             ([0,2,6,4],[0,0,-1]), ([1,5,7,3],[0,0,1])]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(path):
    path = path.resolve()
    return str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)


def bundle_episode(record_path, output):
    episode = json.loads(record_path.read_text())
    observations = output / "observations"
    observations.mkdir(exist_ok=True)
    images = []
    for source in sorted(record_path.parent.glob("message_*.png")):
        target = observations / source.name
        shutil.copyfile(source, target)
        images.append({"source": source_path(source), "asset": f"observations/{source.name}",
                       "sha256": sha256(source), "byte_identical": sha256(source) == sha256(target)})
    names = {Path(item["asset"]).name: item["asset"] for item in images}
    def rebase(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "image" and isinstance(item, str):
                    value[key] = names[Path(item).name]
                else:
                    rebase(item)
        elif isinstance(value, list):
            for item in value:
                rebase(item)
    rebase(episode)
    (output / "episode.json").write_text(json.dumps(episode, indent=2, ensure_ascii=False) + "\n")
    return images


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def view_height(bounds, margin=1.08):
    return float(max(bounds[3] - bounds[1], (bounds[2] - bounds[0]) / (WIDTH / HEIGHT)) * margin)


def camera(t, bounds, zoom, bbox, z, azimuth, maximum_forward_shift=.28, inspection_view_height=None):
    # Both BEV stages are strictly vertical, orthographic and north-up.
    global_xy = (bounds[:2] + bounds[2:]) / 2
    zoom_xy = (zoom[:2] + zoom[2:]) / 2
    prediction_xy = (bbox[:2] + bbox[2:]) / 2
    global_h, zoom_h = view_height(bounds), view_height(zoom)
    zoom_progress = float(smooth((t - ZOOM_START) / ZOOM_DURATION))
    target_xy = (1 - zoom_progress) * global_xy + zoom_progress * zoom_xy
    height = math.exp((1 - zoom_progress) * math.log(global_h) + zoom_progress * math.log(zoom_h))
    target = np.array([*target_xy, z])
    distance, elevation, up, projection = 20 * height, 90.0, np.array([0., 1., 0.]), "orthographic"
    if t >= OBLIQUE_START:
        tilt = float(smooth((t - OBLIQUE_START) / OBLIQUE_DURATION))
        final_h = inspection_view_height or max(float(np.max(bbox[2:] - bbox[:2])) * 1.65, 20.0)
        height = math.exp((1 - tilt) * math.log(zoom_h) + tilt * math.log(final_h))
        lateral = np.array([-math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))])
        corners = np.array([[x, y] for x in (bounds[0], bounds[2]) for y in (bounds[1], bounds[3])])
        limits = corners @ lateral
        margin = min(final_h * WIDTH / HEIGHT / 2 * .75, (limits.max()-limits.min())*.45)
        center = float(prediction_xy @ lateral)
        adjusted = float(np.clip(center, limits.min() + margin, limits.max() - margin))
        inspection_xy = prediction_xy + lateral * (adjusted - center)
        # Near a scan boundary, keep more measured context in front of the
        # camera. Cap the adjustment so the actual prediction stays in view.
        forward = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
        forward_limits = corners @ forward
        forward_center = float(prediction_xy @ forward)
        forward_margin = min(final_h * .65, (forward_limits.max()-forward_limits.min())*.45)
        desired = float(np.clip(forward_center, forward_limits.min()+forward_margin,
                                forward_limits.max()-forward_margin))
        shift = float(np.clip(desired-forward_center,-final_h*maximum_forward_shift,final_h*maximum_forward_shift))
        inspection_xy += forward * shift
        target = np.array([*((1 - tilt) * zoom_xy + tilt * inspection_xy), z])
        final_distance = final_h / (2 * math.tan(math.radians(21)))
        distance = math.exp((1 - tilt) * math.log(20 * zoom_h) + tilt * math.log(final_distance))
        elevation = 90 - 60 * tilt
        up = (1 - tilt) * np.array([0., 1., 0.]) + tilt * np.array([0., 0., 1.])
        projection = "perspective"
    az, el = np.radians([azimuth, elevation])
    # Explicit vertical eye position, rather than rounding an 89-degree view.
    offset = np.array([0., 0., distance]) if elevation == 90 else distance * np.array([
        np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = target + offset
    fov = math.degrees(2 * math.atan(height / (2 * distance)))
    return {"t": t, "eye": eye.tolist(), "target": target.tolist(), "up": up.tolist(),
            "azimuth": azimuth, "elevation": elevation, "distance": distance,
            "view_height": height, "projection": projection, "vertical_fov_degrees": fov}


def gentle_camera(t, bounds, bbox, z, inspection_view_height=None):
    """Pan, zoom, then tilt around a locked target, without image roll.

    The BEV stays north-up. A small azimuth change during the tilt reveals a
    side face while keeping world Z vertical. Tool records stay unchanged.
    """
    global_xy = (bounds[:2] + bounds[2:]) / 2
    prediction_xy = (bbox[:2] + bbox[2:]) / 2
    pan = float(smooth((t - GENTLE["pan_start_seconds"]) /
                       (GENTLE["pan_complete_seconds"] - GENTLE["pan_start_seconds"])))
    zoom = float(smooth((t - GENTLE["zoom_start_seconds"]) /
                        (GENTLE["zoom_complete_seconds"] - GENTLE["zoom_start_seconds"])))
    tilt = float(smooth((t - GENTLE["oblique_start_seconds"]) /
                        (GENTLE["oblique_complete_seconds"] - GENTLE["oblique_start_seconds"])))
    final_h = inspection_view_height or max(float(np.max(bbox[2:] - bbox[:2])) * 2.0, 32.0)
    height = math.exp((1 - zoom) * math.log(view_height(bounds)) + zoom * math.log(final_h))
    target = np.array([*((1 - pan) * global_xy + pan * prediction_xy), z])
    elevation = 90.0 - 60.0 * tilt
    azimuth = 270.0 + GENTLE_AZIMUTH_OFFSET * tilt
    az = math.radians(azimuth)
    angle, distance = math.radians(elevation), 20 * height
    eye = target + distance * np.array([math.cos(angle)*math.cos(az),
                                        math.cos(angle)*math.sin(az), math.sin(angle)])
    if elevation == 90:
        eye[:2] = target[:2]
    up = [-math.sin(angle)*math.cos(az), -math.sin(angle)*math.sin(az), math.cos(angle)] if elevation != 90 else [0., 1., 0.]
    return {"t": t, "eye": eye.tolist(), "target": target.tolist(), "up": up,
            "azimuth": azimuth, "elevation": elevation, "distance": distance,
            "view_height": height, "projection": "orthographic",
            "vertical_fov_degrees": math.degrees(2 * math.atan(height / (2 * distance)))}


def emphasize_prediction(image, bbox, zs, cam, t):
    """Project the same world-space box through the actual render camera.

    Visible faces, solid front edges and faint dashed rear edges distinguish
    depth. All vertices use the recorded XY and measured Z, without padding.
    """
    corners = np.array([[x, y, z, 1.] for x in (bbox[0], bbox[2])
                        for y in (bbox[1], bbox[3]) for z in zs])
    view = np.asarray(cam.get_view_matrix())
    matrix = np.asarray(cam.get_projection_matrix()) @ view
    clip = corners @ matrix.T
    ndc = clip[:, :3] / clip[:, 3:4]
    screen = (ndc[:, :2] * [1., -1.] + 1.) * [WIDTH / 2, HEIGHT / 2]
    toward_eye = np.linalg.inv(view)[:3,3] - corners[:,:3].mean(axis=0)
    visible_faces = [face for face,normal in BOX_FACES if np.dot(normal,toward_eye)>1e-6]
    visible_edges = {tuple(sorted((a,b))) for face in visible_faces
                     for a,b in zip(face,face[1:]+face[:1])}
    scale = 2
    layer = Image.new("RGBA", (WIDTH*scale, HEIGHT*scale))
    draw = ImageDraw.Draw(layer)
    points = screen * scale
    for face in visible_faces:
        # Keep scene details visible through gently shaded box faces.
        alpha = 22 if all(corners[i,2]==zs[1] for i in face) else 14
        draw.polygon([tuple(points[i]) for i in face], fill=(255,75,75,alpha))
    for a,b in BOX_EDGES:
        if (a,b) in visible_edges:
            continue
        start,end = points[a],points[b]
        length = float(np.linalg.norm(end-start))
        if length < 1e-6:
            continue
        # Rear edges provide depth cues instead of competing with the outline.
        for offset in np.arange(0,length,18*scale):
            p=start+(end-start)*offset/length
            q=start+(end-start)*min(offset+10*scale,length)/length
            draw.line([tuple(p),tuple(q)],fill=(196,89,85,160),width=3*scale)
    for width,color in ((9,(18,24,32,220)),(6,(255,75,75,255))):
        for a,b in sorted(visible_edges):
            draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=width*scale)
    layer = layer.resize(image.size,Image.Resampling.LANCZOS)
    alpha = float(smooth((t-GENTLE["prediction_start_seconds"])/PREDICTION_FADE_SECONDS))
    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda value: round(value*alpha)))
    return Image.alpha_composite(image.convert("RGBA"),layer).convert("RGB")


def voxel_keys(points, spacing):
    cells = np.floor(points / spacing).astype(np.int64)
    if np.any(cells < -(1 << 20)) or np.any(cells >= (1 << 20)):
        raise ValueError("Point-cloud coordinates exceed voxel key range")
    cells += 1 << 20
    return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]


def inside_xy(points, bounds):
    return ((points[:, 0] >= bounds[0]) & (points[:, 1] >= bounds[1]) &
            (points[:, 0] <= bounds[2]) & (points[:, 1] <= bounds[3]))


def load_cloud(source, cache, bbox, zoom, height_percentiles=(.02, .95)):
    size = float(np.max(bbox[2:] - bbox[:2]))
    small_object = size < 10
    height_stride = 1 if small_object else 3
    zoom_center = (zoom[:2] + zoom[2:]) / 2
    zoom_half = np.array([WIDTH / HEIGHT, 1.]) * view_height(zoom) * .575
    final_center = (bbox[:2] + bbox[2:]) / 2
    final_half = max(size * 1.65, 18.)
    dense_bounds = np.r_[np.minimum(zoom_center - zoom_half, final_center - final_half),
                         np.maximum(zoom_center + zoom_half, final_center + final_half)]
    padding = 25. if small_object else max(10., size * .2)
    detail_bounds = bbox + np.array([-padding, -padding, padding, padding])
    spacings = [.015, .05, .30] if small_object else [.055, .07, .40]
    sampling = {"strategy": "spatial voxel representatives; one original point per cell and tier",
                "detail_world_xy": detail_bounds.tolist(), "dense_world_xy": dense_bounds.tolist(),
                "voxel_metres": dict(zip(("detail", "dense", "context"), spacings)),
                "height_sample_stride": height_stride,
                "display_height_percentiles": list(height_percentiles),
                "global_point_cap": None}
    stat = source.stat()
    signature = {"path": str(source.resolve()), "bytes": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "prediction_xy": bbox.tolist(),
                 "sampling": sampling}
    manifest = cache.with_suffix(".json")
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature:
        with np.load(cache) as data:
            return data["xyz"], data["rgb"], data["z_extent"], int(data["source_vertices"]), sampling
    vertices = PlyData.read(source, mmap=True)["vertex"].data
    tier_keys, tier_indices, heights = [[], [], []], [[], [], []], []
    # Spatial sampling covers the actual camera window. Keep measured points
    # rather than averaging cells, and never globally thin the fine detail tier.
    for begin in range(0, len(vertices), 2_000_000):
        block = vertices[begin:begin + 2_000_000]
        points = np.column_stack([block[n] for n in ("x", "y", "z")])
        detail = inside_xy(points, detail_bounds)
        dense = inside_xy(points, dense_bounds) & ~detail
        for tier, mask in enumerate((detail, dense, ~(detail | dense))):
            local_indices = np.flatnonzero(mask)
            keys, selected = np.unique(voxel_keys(points[mask], spacings[tier]), return_index=True)
            tier_keys[tier].append(keys)
            tier_indices[tier].append(local_indices[selected] + begin)
        # Preserve the earlier display height and framing for comparison.
        local = block[::height_stride]
        target = ((local["x"] >= bbox[0]) & (local["x"] <= bbox[2]) &
                  (local["y"] >= bbox[1]) & (local["y"] <= bbox[3]))
        heights.append(local["z"][target])
        if begin % 10_000_000 == 0:
            print(f"spatial sampling {begin}/{len(vertices)}", flush=True)
    indices, tier_counts = [], {}
    for tier, name in enumerate(("detail", "dense", "context")):
        keys = np.concatenate(tier_keys[tier])
        values = np.concatenate(tier_indices[tier])
        _, selected = np.unique(keys, return_index=True)
        indices.append(values[selected])
        tier_counts[name] = len(selected)
    source_indices = np.sort(np.concatenate(indices))
    selected = vertices[source_indices]
    xyz = np.column_stack([selected[n] for n in ("x", "y", "z")])
    rgb = np.column_stack([selected[n] for n in ("red", "green", "blue")])
    target_z = np.concatenate(heights)
    if not len(target_z):
        raise ValueError("No measured points inside the prediction; cannot derive display height")
    z_extent = np.quantile(target_z, height_percentiles)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xyz=xyz, rgb=rgb, source_indices=source_indices,
             z_extent=z_extent, source_vertices=len(vertices))
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    print(json.dumps({"spatial_sampling_tier_counts": tier_counts}), flush=True)
    return xyz, rgb, z_extent, len(vertices), sampling


def add_point_geometry(renderer, name, xyz, rgb, material):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector((rgb.astype(float) / 255) ** 2.2)
    renderer.scene.add_geometry(name, cloud, material, False)


def add_cloud_tiles(renderer, xyz, rgb, material):
    # Small geometry bounds let the renderer discard off-screen city tiles.
    cells = np.floor(xyz[:, :2] / 40).astype(np.int64)
    keys = ((cells[:, 0] + (1 << 20)) << 21) | (cells[:, 1] + (1 << 20))
    order = np.argsort(keys, kind="stable")
    ordered_keys = keys[order]
    cuts = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1, len(order)]
    names = []
    for index, (begin, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        selected = order[begin:end]
        name = f"points_{index}"
        add_point_geometry(renderer, name, xyz[selected], rgb[selected], material)
        names.append(name)
    _, overview = np.unique(voxel_keys(xyz, .5), return_index=True)
    add_point_geometry(renderer, "overview", xyz[overview], rgb[overview], material)
    return names, len(overview)


def close_surface_gaps(image, depth, radius, render_scale):
    """Interpolate only missing coverage surrounded by consistent-depth samples.

    Existing geometry pixels, including true dark RGB, are never overwritten.
    The closing mask excludes large voids and convex silhouette boundaries.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if radius <= 0 or not np.any(valid):
        return image, 0
    r = radius * render_scale
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    candidate = closed.astype(bool) & ~valid
    if not np.any(candidate):
        return image, 0
    distance, nearest = ndimage.distance_transform_edt(~valid, return_indices=True)
    low = cv2.erode(np.where(valid, depth, np.float32(1e10)), kernel)
    high = cv2.dilate(np.where(valid, depth, np.float32(0)), kernel)
    donor_depth = depth[nearest[0], nearest[1]]
    fill = candidate & (distance <= r) & ((high - low) <= (.20 + .0075 * donor_depth))
    result = image.copy()
    result[fill] = image[nearest[0][fill], nearest[1][fill]]
    return result, int(np.count_nonzero(fill))


def box_mesh(bbox, zs):
    radius = float(np.clip(np.max(bbox[2:] - bbox[:2]) * 0.0036, 0.018, 0.20))
    corners = np.array([[x, y, z] for x in [bbox[0], bbox[2]]
                        for y in [bbox[1], bbox[3]] for z in zs])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    mesh = o3d.geometry.TriangleMesh()
    for a, b in edges:
        direction = corners[b] - corners[a]
        length = np.linalg.norm(direction)
        if length <= 1e-6:
            continue  # A perfectly planar surface has no vertical edges.
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
        z_axis, axis = np.array([0., 0., 1.]), direction / length
        cross = np.cross(z_axis, axis)
        if np.linalg.norm(cross) > 1e-8:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(cross / np.linalg.norm(cross) * np.arccos(np.clip(z_axis @ axis, -1, 1)))
            cylinder.rotate(rotation, center=(0, 0, 0))
        cylinder.translate((corners[a] + corners[b]) / 2)
        mesh += cylinder
    mesh.paint_uniform_color(GOLD)
    return mesh


def overlay(image, t, iou, scene, gentle=False):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = WIDTH / 960
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", round(15 * scale))
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", round(17 * scale))
    draw.text((round(28 * scale), round(23 * scale)), "URBEX", font=bold, fill="#f4f5f7", stroke_width=1, stroke_fill="#09111c")
    phase = ("Global BEV" if t < GENTLE["zoom_start_seconds"] else
             "Zoomed BEV" if t < GENTLE["oblique_start_seconds"] else "Oblique view") if gentle else (
                 "Global BEV" if t < 2 else "Zoomed BEV" if t < OBLIQUE_START else "Oblique view")
    draw.text((WIDTH - round(28 * scale), round(25 * scale)), phase, anchor="ra", font=font, fill="#e2e6ec", stroke_width=1, stroke_fill="#09111c")
    draw.text((round(28 * scale), HEIGHT - round(30 * scale)), f"{scene}  /  3D replay", font=font, fill="#bcc5d0", stroke_width=1, stroke_fill="#09111c")
    if t >= (GENTLE["prediction_start_seconds"] if gentle else OBLIQUE_END):
        draw.text((WIDTH - round(28 * scale), HEIGHT - round(30 * scale)), f"Prediction  ·  IoU {iou:.3f}", anchor="ra", font=font, fill="#ff8e85" if gentle else "#ffbc58", stroke_width=1, stroke_fill="#09111c")
    # A short dissolve at the loop boundary, instead of a sudden camera jump.
    duration = 12.0 if gentle else SECONDS
    alpha = min(float(smooth(t / LOOP_FADE_SECONDS)), float(smooth((duration - t) / LOOP_FADE_SECONDS)))
    if alpha < 1:
        image = Image.blend(Image.new("RGB", image.size, tuple(round(v * 255) for v in BACKGROUND[:3])), image, alpha)
    return image


def encode_gif(video, output, width, fps, colors):
    # Round down to sample through the final fade when reducing the frame rate.
    filters = (f"fps={fps}:round=down,scale={width}:-1:flags=lanczos,split[a][b];"
               f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
               "[b][p]paletteuse=dither=none")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-threads", "2", "-i", str(video), "-filter_complex_threads", "1",
                    "-filter_complex", filters, "-loop", "0",
                    "-final_delay", str(max(1, round(100 / fps))),
                    str(output)], check=True)
    encoding = {"optimizer": "ffmpeg", "lossy": 0}
    optimizer = shutil.which("gifsicle")
    if optimizer:
        compact = output.with_name(f".{output.stem}.compact.gif")
        try:
            subprocess.run([optimizer, "--optimize=3", "--threads=2",
                            f"--lossy={GIF_LOSSY}", str(output), "--output",
                            str(compact)], check=True)
            if compact.stat().st_size < output.stat().st_size:
                compact.replace(output)
                encoding = {"optimizer": "gifsicle", "lossy": GIF_LOSSY}
        finally:
            compact.unlink(missing_ok=True)
    return encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", type=Path, help="Defaults to the recorded scene's release PLY")
    parser.add_argument("--bev-meta", type=Path, help="Defaults to the scene's release BEV mapping")
    parser.add_argument("--record", type=Path, required=True, help="Recorded agent episode JSON")
    parser.add_argument("--work-dir", type=Path, help="Render cache; defaults to <output>/.render-cache")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated replay assets")
    parser.add_argument("--preview", action="store_true", help="Render five keyframes only")
    parser.add_argument("--no-gif", action="store_true", help="Produce MP4/poster only for additional examples")
    parser.add_argument("--camera-motion", choices=("legacy", "gentle"), default="legacy",
                        help="Gentle: separate pan/zoom/tilt, south-side orthographic view, clearer box")
    parser.add_argument("--category", choices=("Building", "Car", "Parking", "Ground"),
                        help="Original annotation category, used only for display geometry")
    parser.add_argument("--presentation-direction", choices=("east", "north", "west", "south"),
                        help="Editorial camera angle; the actual tool direction remains recorded separately")
    parser.add_argument("--inspection-view-height", type=float,
                        help="Editorial final-view height in world metres, without changing the prediction")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point diameter in output pixels")
    parser.add_argument("--render-scale", type=int, default=2, choices=(1, 2), help="Supersampling factor before downscaling")
    parser.add_argument("--gap-radius", type=int, default=3, choices=range(5), help="Maximum depth-guided gap interpolation radius in output pixels; 0 disables it")
    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = args.output / ".render-cache"
    gentle = args.camera_motion == "gentle"
    seconds = 12.0 if gentle else SECONDS
    frame_count = round(seconds * FPS)
    poster_frame = 224 if gentle else POSTER_FRAME
    prediction_start = GENTLE["prediction_start_seconds"] if gentle else PREDICTION_START
    fine_start = GENTLE["zoom_start_seconds"] if gentle else 2.5
    gif_width, gif_fps, gif_colors = GIF_WIDTH, GIF_FPS, GIF_COLORS
    gif_file = "demo-preview.gif" if gentle else GIF_FILE
    if not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("--point-size must be finite and positive")
    if args.inspection_view_height is not None and (not math.isfinite(args.inspection_view_height) or args.inspection_view_height<=0):
        parser.error("--inspection-view-height must be finite and positive")
    record = json.loads(args.record.read_text())
    scene, dataset = record["scene_id"], record["dataset"]
    if dataset not in ("stpls3d", "cityrefer"):
        raise ValueError("Expected a released STPLS3D or CityRefer scene")
    if args.point_cloud is None:
        candidates = ([WORKSPACE / f"raw_data/STPLS3D/RealWorldData/{scene}{suffix}.ply" for suffix in ("", "_points")]
                      if dataset == "stpls3d" else [WORKSPACE / f"raw_data/sensaturban/ply/{split}/{scene}.ply" for split in ("train", "test")])
        args.point_cloud = next((path for path in candidates if path.exists()), None)
        if args.point_cloud is None:
            raise FileNotFoundError(f"No released point cloud for {scene}")
    if args.bev_meta is None:
        args.bev_meta = WORKSPACE / (f"data/stpls3d/bev_stpls3d/{scene}.json" if dataset == "stpls3d" else f"data/cityrefer/bev_cityrefer/{scene}.json")
    meta = json.loads(args.bev_meta.read_text())
    bounds = np.array([meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"]])
    zoom_actions = [a for a in record["actions"] if a.get("info", {}).get("tool_name") == "zoom_in"]
    render_actions = [a for a in record["actions"] if a.get("info", {}).get("tool_name") == "render_oblique"]
    if not render_actions:
        raise ValueError("The replay requires an actual recorded oblique inspection")
    # Longer episodes can revisit a region. Replay their final recorded views;
    # the bundled episode retains every intermediate tool call and observation.
    # Some policies inspect a region directly from the global view. In that
    # case the editorial top-down approach uses the actual oblique request's
    # footprint. Preserve the absent zoom call explicitly in the provenance.
    focus_action = zoom_actions[-1] if zoom_actions else render_actions[-1]
    zoom = np.array(focus_action["info"]["world_bbox"], dtype=float)
    direction = render_actions[-1]["info"]["direction"]
    if gentle and args.presentation_direction not in (None, "south"):
        parser.error("--camera-motion gentle requires a south-side presentation camera to avoid roll")
    presentation_direction = "south" if gentle else (args.presentation_direction or direction)
    azimuth = {"east": 0, "north": 90, "west": 180, "south": 270}[presentation_direction]
    bbox = np.array(record["world_bbox"], dtype=float)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    height_percentiles = (.02, .30) if args.category in ("Ground", "Parking") else (.02, .95)
    # Keep broad lawns and pitches prominent even near scan boundaries. A
    # large context-preserving shift can otherwise push the actual ground
    # target into the background of the final oblique frame.
    maximum_forward_shift = .08 if args.category == "Ground" else .28
    xyz, rgb, zs, count, sampling = load_cloud(args.point_cloud, args.work_dir / "sampled_scene.npz", bbox, zoom, height_percentiles)
    look_z = float(np.mean(zs))
    print(json.dumps({"measured_vertices": count, "rendered_vertices": len(xyz), "display_z_extent": zs.tolist()}), flush=True)
    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH * args.render_scale, HEIGHT * args.render_scale)
    renderer.scene.set_background(BACKGROUND)
    renderer.scene.show_skybox(False)
    renderer.scene.view.set_color_grading(o3d.visualization.rendering.ColorGrading(
        o3d.visualization.rendering.ColorGrading.Quality.HIGH,
        o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR))
    material = o3d.visualization.rendering.MaterialRecord()
    # Fine point footprints with denser spatial coverage, rather than large dots.
    material.shader, material.point_size = "defaultUnlit", args.point_size * args.render_scale
    tile_names, overview_count = add_cloud_tiles(renderer, xyz, rgb, material)
    highlight = o3d.visualization.rendering.MaterialRecord()
    highlight.shader = "defaultUnlit"
    renderer.scene.add_geometry("prediction", box_mesh(bbox, zs), highlight)
    times = ([.4, 2.3, 6.9, 8.9, 11.2] if gentle else [0.7, 3.5, 4.2, 5.7, 7.2]) if args.preview else [i / FPS for i in range(frame_count)]
    frame_dir = args.work_dir / ("preview" if args.preview else "frames")
    frame_dir.mkdir(exist_ok=True)
    cameras = []
    coverage = []
    last_key, last_image, filled_pixels = None, None, 0
    for index, t in enumerate(times):
        params = gentle_camera(t, bounds, bbox, look_z, args.inspection_view_height) if gentle else camera(t, bounds, zoom, bbox, look_z, azimuth, maximum_forward_shift, args.inspection_view_height)
        fine = t >= fine_start
        renderer.scene.show_geometry("overview", not fine)
        for name in tile_names:
            renderer.scene.show_geometry(name, fine)
        renderer.scene.show_geometry("prediction", not gentle and t >= prediction_start)
        cam = renderer.scene.camera
        cam.look_at(params["target"], params["eye"], params["up"])
        extent = float(np.linalg.norm(bounds[2:] - bounds[:2]) * 2)
        near, far = max(0.05, params["distance"] - extent), params["distance"] + extent
        if params["projection"] == "orthographic":
            half_h = params["view_height"] / 2
            half_w = half_h * WIDTH / HEIGHT
            cam.set_projection(o3d.visualization.rendering.Camera.Projection.Ortho, -half_w, half_w, -half_h, half_h, near, far)
        else:
            cam.set_projection(params["vertical_fov_degrees"], WIDTH / HEIGHT, near, far, o3d.visualization.rendering.Camera.FovType.Vertical)
        key = json.dumps({key: value for key, value in params.items() if key != "t"}) + str((fine, t >= prediction_start))
        if key != last_key:
            pixels = np.asarray(renderer.render_to_image()).copy()
            filled_pixels = 0
            if fine and args.gap_radius:
                depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
                pixels, filled_pixels = close_surface_gaps(pixels, depth, args.gap_radius, args.render_scale)
            last_image = Image.fromarray(pixels)
            if args.render_scale > 1:
                last_image = last_image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            last_key = key
        rendered = last_image.copy()
        if gentle and t >= prediction_start:
            rendered = emphasize_prediction(rendered, bbox, zs, cam, t)
        rendered = overlay(rendered, t, record["iou_2d"], scene, gentle)
        rendered.save(frame_dir / f"{index:04d}.png")
        cameras.append(params)
        coverage.append({"t": t, "level": "fine" if fine else "overview", "interpolated_render_pixels": filled_pixels})
        if index % 20 == 0:
            print(f"rendered {index + 1}/{len(times)}", flush=True)
    del renderer
    if args.preview:
        print(str(frame_dir), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(frame_dir / "%04d.png"), "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output / "demo.mp4")], check=True)
    gif_encoding = None
    if not args.no_gif:
        gif_encoding = encode_gif(args.output / "demo.mp4", args.output / gif_file,
                                  gif_width, gif_fps, gif_colors)
    with Image.open(frame_dir / f"{poster_frame:04d}.png") as poster:
        poster.save(args.output / "poster.png")
    observations = bundle_episode(args.record, args.output)
    provenance = {
        "kind": "raw_point_cloud_3d_camera_replay", "scene": scene, "dataset": dataset,
        "object_category": args.category,
        "display_height_percentiles": list(height_percentiles),
        "point_cloud": source_path(args.point_cloud),
        "point_cloud_sha256": sha256(args.point_cloud),
        "source_vertex_count": count, "rendered_vertex_count": len(xyz),
        "vertex_count_scope": f"total fine geometry uploaded; overview is used before {fine_start:g} seconds and off-screen tiles are culled",
        "record": source_path(args.record), "record_sha256": sha256(args.record),
        "episode_asset": "episode.json", "original_observations": observations,
        "predicted_world_xy": bbox.tolist(), "display_z_extent": zs.tolist(),
        "recorded_zoom_world_xy": zoom.tolist() if zoom_actions else None,
        "focused_world_xy": zoom.tolist(),
        "focus_extent_source": "recorded zoom_in" if zoom_actions else "recorded render_oblique request; no zoom_in call",
        "global_world_xy": bounds.tolist(),
        "recorded_oblique_direction": direction,
        "presentation_oblique_direction": presentation_direction,
        "editorial_inspection_view_height_metres": args.inspection_view_height,
        "editorial_view_override": {"reason": "north-up BEV followed by a slow tilt with a small azimuth change to reveal box depth" if gentle else "show the recorded prediction without foreground-building occlusion",
            "actual_tool_direction": direction, "camera_direction": presentation_direction} if presentation_direction != direction else None,
        "replayed_tool_views": {"zoom_turn": zoom_actions[-1]["turn"] if zoom_actions else None,
            "oblique_turn": render_actions[-1]["turn"],
            "selection": "final recorded zoom and oblique direction" if zoom_actions else "direct recorded oblique region; top-down camera approach is editorial"},
        "bev_iou": record["iou_2d"],
        "z_source": f"{height_percentiles[0]*100:g}th and {height_percentiles[1]*100:g}th percentiles of measured points inside predicted XY; display geometry, not model-predicted height",
        "ground_truth_geometry_used": False, "rgb_to_linear_exponent": 2.2,
        "tone_mapping": "LINEAR",
        "point_size": args.point_size,
        "point_size_units": "output pixels",
        "render_scale": args.render_scale,
        "render_backend": "nvidia_egl" if "10_nvidia.json" in os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "") else "mesa_software",
        "render_work_dir": source_path(args.work_dir),
        "render_size": [WIDTH * args.render_scale, HEIGHT * args.render_scale],
        "downsample_filter": "LANCZOS" if args.render_scale > 1 else None,
        "sampling": sampling,
        "overview_vertex_count": overview_count,
        "overview_voxel_metres": .5, "fine_level_from_seconds": fine_start,
        "geometry_tile_metres": 40, "geometry_tile_count": len(tile_names),
        "surface_gap_interpolation": {"radius_output_pixels": args.gap_radius,
            "method": "morphological closing of missing depth; nearest covered color within radius",
            "max_neighbor_depth_range_metres": "0.20 + 0.0075 * camera depth",
            "existing_geometry_pixels_modified": False,
            "large_voids_filled": False},
        "coverage": coverage,
        "hole_reduction": "denser measured-point coverage, fine footprints, supersampling and bounded depth-guided screen interpolation; no mesh reconstruction",
        "camera_path": "global north-up BEV, pan to prediction, locked-center zoom, then south-side 30-degree tilt with a 15-degree azimuth change; orthographic throughout; transitions are editorial" if gentle else "global north-up orthographic BEV, zoomed north-up orthographic BEV, then 30-degree oblique inspection; transitions are editorial",
        "camera_order": ["global_bev", "zoomed_bev", "oblique"],
        "camera_motion": args.camera_motion,
        "camera_motion_revision": GENTLE_REVISION if gentle else 1,
        "presentation_azimuth_offset_degrees": GENTLE_AZIMUTH_OFFSET if gentle else 0.,
        "orthographic_until_seconds": seconds if gentle else OBLIQUE_START,
        "timing": GENTLE if gentle else {"zoom_start_seconds": ZOOM_START,
            "zoom_complete_seconds": ZOOM_START + ZOOM_DURATION,
            "zoom_hold_seconds": ZOOM_HOLD,
            "oblique_start_seconds": OBLIQUE_START,
            "oblique_complete_seconds": OBLIQUE_END,
            "prediction_start_seconds": PREDICTION_START},
        "minimum_inspection_view_height_metres": 32.0 if gentle else 20.0,
        "inspection_border_margin_half_width_fraction": .75,
        "inspection_border_margin_max_scene_span_fraction": .45,
        "inspection_forward_border_margin_view_height_fraction": .65,
        "maximum_forward_focus_shift_view_height_fraction": 0.0 if gentle else maximum_forward_shift,
        "prediction_highlight": {"method": "camera-projected cuboid faces and edges", "line_width_output_pixels": 6,
            "halo_width_output_pixels": 9, "color": "#ff4b4b", "halo_color": "#121820",
            "rear_edges": "faint dashed", "rear_line_width_output_pixels": 3,
            "visible_face_alpha": {"top":22,"side":14}, "fade_in_seconds": PREDICTION_FADE_SECONDS,
            "geometry_padding_metres": 0., "world_xy_source": "recorded model prediction", "z_source": "measured point percentiles"} if gentle else {"method": "world-space cylinder mesh"},
        "poster_frame": poster_frame,
        "cameras": cameras, "duration_seconds": seconds,
        "mp4_size": [WIDTH, HEIGHT], "mp4_fps": FPS,
        "gif_fps": gif_fps, "gif_width": gif_width,
        "gif_palette_colors": gif_colors, "gif_dither": "none", "gif_loop": "infinite",
        "gif_encoding": gif_encoding,
        "loop_transition": {"method": "fade through background", "fade_seconds": LOOP_FADE_SECONDS},
        "gif_file": None if args.no_gif else gif_file, "mp4_file": "demo.mp4",
        "generator": "code/scripts/demo/build_3d_demo.py",
    }
    (args.output / "3d_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"mp4_bytes": (args.output / "demo.mp4").stat().st_size, "gif_bytes": None if args.no_gif else (args.output / gif_file).stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
