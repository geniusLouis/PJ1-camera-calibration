"""Chapter 4: image undistortion and chessboard line straightness.

The experiment keeps the original image size and camera matrix so that the
original and corrected corner coordinates share one pixel coordinate system.
Run from any directory with ``python code/undistort.py``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = CODE_DIR / "results"
CORNER_PATH = RESULTS_DIR / "corners" / "corners.npz"
CALIBRATION_PATH = RESULTS_DIR / "calibration" / "calibration_30mm.json"
OUTPUT_DIR = RESULTS_DIR / "undistortion"


def read_image(path: Path) -> np.ndarray:
    """Read an image without relying on the platform's Unicode path support."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write an image through OpenCV's encoded representation."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def undistort_image(
    image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """Undistort an image while preserving its size and camera matrix."""
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    new_camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Transform distorted pixel points into the corrected pixel coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    projection = camera_matrix if new_camera_matrix is None else new_camera_matrix
    corrected = cv2.undistortPoints(
        points, camera_matrix, dist_coeffs, P=projection
    )
    return corrected.reshape(-1, 2)


def fit_line_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D orthogonal line and return a point on it and a unit normal."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("A finite line fit needs at least two points.")
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)
    return center, normal


def line_distances(points: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return absolute perpendicular distances from points to a fitted line."""
    center, normal = line
    return np.abs((np.asarray(points, dtype=np.float64) - center) @ normal)


def linearity_rows(
    points: np.ndarray, image_name: str, pattern_size: tuple[int, int], stage: str
) -> list[dict[str, object]]:
    """Compute one row of statistics for every chessboard row and column."""
    columns, rows = pattern_size
    points = np.asarray(points, dtype=np.float64).reshape(rows, columns, 2)
    output: list[dict[str, object]] = []
    for direction in ("horizontal", "vertical"):
        count = rows if direction == "horizontal" else columns
        for line_index in range(count):
            line_points = points[line_index, :, :] if direction == "horizontal" else points[:, line_index, :]
            distances = line_distances(line_points, fit_line_pca(line_points))
            output.append(
                {
                    "image_name": image_name,
                    "stage": stage,
                    "direction": direction,
                    "line_index": line_index,
                    "point_count": len(line_points),
                    "rmse_px": float(np.sqrt(np.mean(distances**2))),
                    "max_distance_px": float(np.max(distances)),
                }
            )
    return output


def summarize_linearity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate line RMSEs by image and direction for before/after comparison."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["image_name"]), str(row["stage"]), str(row["direction"]))
        groups.setdefault(key, []).append(float(row["rmse_px"]))
    by_key = {
        key: float(np.sqrt(np.mean(np.square(values))))
        for key, values in groups.items()
    }
    names = sorted({key[0] for key in groups})
    output = []
    for name in names:
        for direction in ("horizontal", "vertical", "all"):
            before = [by_key[(name, "before", d)] for d in ("horizontal", "vertical")] if direction == "all" else [by_key[(name, "before", direction)]]
            after = [by_key[(name, "after", d)] for d in ("horizontal", "vertical")] if direction == "all" else [by_key[(name, "after", direction)]]
            before_value = float(np.sqrt(np.mean(np.square(before))))
            after_value = float(np.sqrt(np.mean(np.square(after))))
            output.append(
                {
                    "image_name": name,
                    "direction": direction,
                    "before_rmse_px": before_value,
                    "after_rmse_px": after_value,
                    "change_px": after_value - before_value,
                    "relative_change_percent": (after_value - before_value) / before_value * 100.0,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_side_by_side(original: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    """Return an annotated original/corrected comparison image."""
    canvas = np.hstack((original, corrected))
    cv2.putText(canvas, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Undistorted", (original.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2, cv2.LINE_AA)
    return canvas


def make_local_comparison(
    original: np.ndarray, corrected: np.ndarray, original_center: np.ndarray, corrected_center: np.ndarray
) -> np.ndarray:
    """Crop the same physical board area around corresponding center points."""
    radius = 70
    patches = []
    for image, center in ((original, original_center), (corrected, corrected_center)):
        x, y = np.rint(center).astype(int)
        padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        x += radius
        y += radius
        patch = padded[y - radius:y + radius + 1, x - radius:x + radius + 1]
        patches.append(patch)
    canvas = np.hstack(patches)
    cv2.putText(canvas, "Original", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Undistorted", (2 * radius + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 150, 0), 1, cv2.LINE_AA)
    return canvas


def make_linearity_figure(
    original: np.ndarray, corrected: np.ndarray, original_points: np.ndarray, corrected_points: np.ndarray, name: str, output: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns, rows = 11, 7
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, image, points, title in zip(axes, (original, corrected), (original_points, corrected_points), ("Original", "Undistorted")):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        grid = points.reshape(rows, columns, 2)
        for line in list(grid) + list(grid.transpose(1, 0, 2)):
            center, normal = fit_line_pca(line)
            direction = np.array([-normal[1], normal[0]])
            extent = np.max(np.abs((line - center) @ direction)) + 15
            ends = np.array([center - extent * direction, center + extent * direction])
            ax.plot(ends[:, 0], ends[:, 1], color="#00d4ff", linewidth=0.8, alpha=.75)
        ax.scatter(points[:, 0], points[:, 1], s=8, c="#ffcc33")
        ax.set_title(title)
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    fig.suptitle(f"{name}: fitted chessboard lines")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    with np.load(CORNER_PATH, allow_pickle=False) as corners:
        names = [str(name) for name in corners["image_names"].tolist()]
        width, height = (int(value) for value in corners["image_size"])
        pattern_size = tuple(int(value) for value in corners["pattern_size"])
        observed = corners["image_points_refined"].astype(np.float64).reshape(len(names), -1, 2)
    if calibration["image_names"] != names or calibration["image_size"] != [width, height]:
        raise ValueError("Calibration and corner metadata do not match.")
    if observed.shape[1] != pattern_size[0] * pattern_size[1]:
        raise ValueError("Unexpected number of corner points.")

    camera_matrix = np.asarray(calibration["K"], dtype=np.float64)
    dist_coeffs = np.asarray(calibration["dist_coeffs"], dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("Invalid camera matrix.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    point_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for name, points in zip(names, observed, strict=True):
        original = read_image(DATA_DIR / name)
        if original.shape[:2] != (height, width):
            raise ValueError(f"Unexpected image dimensions for {name}.")
        corrected = undistort_image(original, camera_matrix, dist_coeffs)
        corrected_points = undistort_points(points, camera_matrix, dist_coeffs, camera_matrix)
        for index, (before, after) in enumerate(zip(points, corrected_points, strict=True)):
            row, column = divmod(index, pattern_size[0])
            point_rows.append({"image_name": name, "point_index": index, "board_row": row, "board_col": column, "original_u_px": float(before[0]), "original_v_px": float(before[1]), "undistorted_u_px": float(after[0]), "undistorted_v_px": float(after[1])})
        line_rows.extend(linearity_rows(points, name, pattern_size, "before"))
        line_rows.extend(linearity_rows(corrected_points, name, pattern_size, "after"))
        center_index = (pattern_size[1] // 2) * pattern_size[0] + pattern_size[0] // 2
        save_image(figures / f"{Path(name).stem}_undistorted.png", corrected)
        save_image(figures / f"{Path(name).stem}_comparison.png", make_side_by_side(original, corrected))
        save_image(figures / f"{Path(name).stem}_local.png", make_local_comparison(original, corrected, points[center_index], corrected_points[center_index]))
        make_linearity_figure(original, corrected, points, corrected_points, Path(name).stem, figures / f"{Path(name).stem}_lines.png")

    summary_rows = summarize_linearity(line_rows)
    write_csv(output_dir / "points.csv", point_rows)
    write_csv(output_dir / "linearity.csv", line_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    metadata = {
        "schema_version": 1,
        "image_names": names,
        "image_size": [width, height],
        "pattern_size": list(pattern_size),
        "calibration_file": str(CALIBRATION_PATH.relative_to(CODE_DIR.parent)),
        "corners_file": str(CORNER_PATH.relative_to(CODE_DIR.parent)),
        "K": camera_matrix.tolist(),
        "new_K": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "distortion_order": calibration["distortion_order"],
        "crop": False,
        "coordinate_transform": "cv2.undistortPoints(points, K, dist_coeffs, P=K)",
        "line_fit": "orthogonal least squares by PCA; all 7 rows and 11 columns",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(names)} images and {len(point_rows)} corners: {output_dir}")
    for row in summary_rows:
        if row["direction"] == "all":
            print(f"{row['image_name']}: before {row['before_rmse_px']:.6f} px, after {row['after_rmse_px']:.6f} px")
    return metadata


if __name__ == "__main__":
    run()
