"""Run the Chapter 5 comparison on the four saved 30 mm views."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate import BASE_FLAGS, calibrate
from evaluate import RMSE_TOLERANCE_PX, reprojection_errors, summarize
from undistort import (
    CORNER_PATH,
    DATA_DIR,
    RESULTS_DIR,
    fit_line_pca,
    linearity_rows,
    read_image,
    save_image,
    summarize_linearity,
    undistort_image,
    undistort_points,
    write_csv,
)


MODELS = (
    ("M0", (), BASE_FLAGS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2),
    ("M1", ("k1", "k2"), BASE_FLAGS),
    ("M2", ("k1", "k2", "p1", "p2"), cv2.CALIB_FIX_K3),
)
DISTORTION_ORDER = ("k1", "k2", "p1", "p2", "k3")
SUBSET_MODEL = "M1"
COMPARISON_IMAGE = "Im_L_4.png"
OUTPUT_DIR = RESULTS_DIR / "model_comparison"
SERIES = (
    ("Original / M0", (91, 81, 71), cv2.MARKER_SQUARE),
    ("M1", (153, 114, 0), cv2.MARKER_DIAMOND),
    ("M2", (32, 71, 208), cv2.MARKER_TRIANGLE_UP),
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def selected_lines(points, pattern):
    """Use the top row and left column, both near the image edge."""
    columns, rows = pattern
    grid = points.reshape(rows, columns, 2)
    return grid[0, :], grid[:, 0]


def signed_residuals(points, axis):
    center, normal = fit_line_pca(points)
    if normal[axis] < 0:
        normal = -normal
    return (points - center) @ normal


def straightness_panel(rows, names):
    """Show change from the original for each image, with a shared zero."""
    canvas = np.full((580, 1500, 3), 255, dtype=np.uint8)
    ink = (45, 36, 31)
    font = cv2.FONT_HERSHEY_SIMPLEX
    changes = {
        (row["model"], row["image_name"]): row["change_px"]
        for row in rows if row["direction"] == "all" and row["model"] != "M0"
    }
    limit = max(np.ceil(1000 * max(abs(value) for value in changes.values())) / 1000, 0.001)
    zero, span = 794, 490
    cv2.putText(canvas, "Corrected RMSE minus original RMSE (px)", (63, 56), font,
                1.24, ink, 2, cv2.LINE_AA)
    for x, (label, color, _) in zip((1110, 1270), SERIES[1:], strict=True):
        cv2.rectangle(canvas, (x, 32), (x + 30, 54), color, -1)
        cv2.putText(canvas, label, (x + 43, 54), font, 0.94, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Lower error", (392, 120), font, 0.98, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Higher error", (970, 120), font, 0.98, ink, 2, cv2.LINE_AA)
    cv2.line(canvas, (zero, 138), (zero, 528), ink, 3, cv2.LINE_AA)
    cv2.putText(canvas, "Original / M0 = 0", (zero - 150, 563), font,
                0.95, ink, 2, cv2.LINE_AA)
    for index, name in enumerate(names):
        y = 186 + index * 96
        cv2.putText(canvas, Path(name).stem, (63, y + 10), font, 1.04, ink, 2, cv2.LINE_AA)
        for offset, (label, color, _) in zip((-20, 20), SERIES[1:], strict=True):
            change = changes[label, name]
            x = round(zero + change * span / limit)
            cv2.line(canvas, (zero, y + offset), (x, y + offset), color, 13, cv2.LINE_AA)
            label_x = x - 172 if change < 0 else x + 14
            cv2.putText(canvas, f"{change:+.6f}", (label_x, y + offset + 9), font,
                        0.82, ink, 2, cv2.LINE_AA)
    return canvas


def residual_panel(residuals, images):
    """Place actual images above enlarged deviations from the two zero lines."""
    canvas = np.full((790, 1500, 3), 255, dtype=np.uint8)
    photo_strip = np.full((250, 1500, 3), 255, dtype=np.uint8)
    ink = (45, 36, 31)
    guide = (217, 217, 217)
    font = cv2.FONT_HERSHEY_SIMPLEX
    limit = max(np.ceil(10 * max(np.max(np.abs(v)) for v in residuals.values())) / 10, 0.1)
    scale = 80 / limit
    cv2.putText(canvas, "Top row / horizontal reference", (55, 118), font,
                1.12, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Left column / vertical reference", (55, 423), font,
                1.12, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Zero line to frame: {limit:.2f} px", (933, 423), font,
                0.86, ink, 2, cv2.LINE_AA)
    for index, (label, color, marker) in enumerate(SERIES):
        left = index * 500 + 55
        center = left + 194
        row_y = 258
        cv2.putText(canvas, label, (left, 51), font, 1.25, color, 3, cv2.LINE_AA)
        image = images[label]
        photo_width = 400
        photo_height = round(photo_width * image.shape[0] / image.shape[1])
        thumbnail = cv2.resize(image, (photo_width, photo_height), interpolation=cv2.INTER_AREA)
        photo_strip[:photo_height, left:left + photo_width] = thumbnail
        cv2.rectangle(canvas, (left, row_y - 80), (left + 386, row_y + 80), guide, 2)
        cv2.line(canvas, (left, row_y), (left + 386, row_y), ink, 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (center - 80, 465), (center + 80, 741), guide, 2)
        cv2.line(canvas, (center, 465), (center, 741), ink, 2, cv2.LINE_AA)
        row = residuals[label, "row"]
        column = residuals[label, "column"]
        row_x = np.linspace(left + 12, left + 374, len(row))
        column_y = np.linspace(477, 729, len(column))
        row_path = np.rint(np.column_stack((row_x, row_y + row * scale))).astype(np.int32)
        column_path = np.rint(np.column_stack((center + column * scale, column_y))).astype(np.int32)
        for path in (row_path, column_path):
            cv2.polylines(canvas, [path], False, color, 3, cv2.LINE_AA)
            for vertex in path:
                cv2.drawMarker(canvas, tuple(vertex), color, marker, 16, 3, cv2.LINE_AA)
    return np.vstack((canvas[:70], photo_strip, canvas[70:]))


def run(output_dir=OUTPUT_DIR):
    with np.load(CORNER_PATH, allow_pickle=False) as data:
        objects = data["object_points"]
        observations = data["image_points_refined"]
        names = data["image_names"].tolist()
        image_size = tuple(data["image_size"].tolist())
        pattern = tuple(data["pattern_size"].tolist())
        square_mm = float(data["square_size_mm"])
    if len(names) != 4 or square_mm != 30:
        raise ValueError("The comparison requires the four 30 mm views.")

    original = read_image(DATA_DIR / COMPARISON_IMAGE)
    if original.shape[1::-1] != image_size:
        raise ValueError("The comparison image dimensions differ from the saved observations.")

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparison_points = {"Original / M0": observations[names.index(COMPARISON_IMAGE)]}
    comparison_images = {"Original / M0": original}
    parameters, errors, point_errors, lines, straightness = [], [], [], [], []
    before_lines = [
        row
        for name, points in zip(names, observations, strict=True)
        for row in linearity_rows(points, name, pattern, "before")
    ]

    for model, free_coeffs, flags in MODELS:
        rms, K, dist, rvecs, tvecs = calibrate(
            objects, observations, image_size, flags=flags
        )
        coefficients = dist.ravel()
        if len(coefficients) != len(DISTORTION_ORDER):
            raise RuntimeError(f"{model}: unexpected number of distortion coefficients.")
        distortion = dict(zip(DISTORTION_ORDER, coefficients.tolist(), strict=True))
        if any(value != 0 for key, value in distortion.items() if key not in free_coeffs):
            raise RuntimeError(f"{model}: a fixed distortion coefficient is nonzero.")
        parameters.append({
            "model": model, "fx_px": K[0, 0], "fy_px": K[1, 1],
            "cx_px": K[0, 2], "cy_px": K[1, 2], **distortion,
        })

        views = reprojection_errors(objects, observations, K, dist, rvecs, tvecs)
        pooled = np.concatenate([view.distances for view in views])
        total_rmse = summarize(pooled)[1]
        if abs(total_rmse - rms) > RMSE_TOLERANCE_PX:
            raise RuntimeError(f"{model}: recomputed RMSE differs from OpenCV.")
        for name, distances in [*zip(names, (v.distances for v in views)), ("all", pooled)]:
            count, error, maximum = summarize(distances)
            errors.append({
                "model": model, "image_name": name, "point_count": count,
                "rmse_px": error, "max_error_px": maximum,
            })

        after_lines = []
        for name, observed, view in zip(names, observations, views, strict=True):
            pixels = observed.reshape(-1, 2)
            corrected = undistort_points(pixels, K, dist, K)
            if name == COMPARISON_IMAGE and model != "M0":
                comparison_points[model] = corrected
            after_lines.extend(linearity_rows(corrected, name, pattern, "after"))
            for index, (pixel, projection, residual, distance) in enumerate(
                zip(pixels, view.projected, view.residuals, view.distances, strict=True)
            ):
                point_errors.append({
                    "model": model, "image_name": name, "point_index": index,
                    "observed_u_px": float(pixel[0]), "observed_v_px": float(pixel[1]),
                    "projected_u_px": projection[0], "projected_v_px": projection[1],
                    "residual_u_px": residual[0], "residual_v_px": residual[1],
                    "error_px": distance,
                })
        model_lines = before_lines + after_lines
        lines.extend({"model": model, **row} for row in model_lines)
        straightness.extend(
            {"model": model, **row} for row in summarize_linearity(model_lines)
        )

        corrected = undistort_image(original, K, dist)
        save_image(figures / f"{COMPARISON_IMAGE[:-4]}_{model}.png", corrected)
        if model != "M0":
            comparison_images[model] = corrected
        write_json(output_dir / f"calibration_{model}.json", {
            "model": model, "flags": flags, "free_distortion": free_coeffs,
            "square_mm": square_mm, "image_size": image_size, "image_names": names,
            "K": K.tolist(), "distortion_order": DISTORTION_ORDER,
            "dist_coeffs": coefficients.tolist(),
            "rvecs": [rvec.ravel().tolist() for rvec in rvecs],
            "tvecs_mm": [tvec.ravel().tolist() for tvec in tvecs],
            "rmse_px": total_rmse, "opencv_rmse_px": rms,
        })
        print(f"{model}: {total_rmse:.6f} px")

    residuals = {}
    for label, points in comparison_points.items():
        row, column = selected_lines(points, pattern)
        residuals[label, "row"] = signed_residuals(row, axis=1)
        residuals[label, "column"] = signed_residuals(column, axis=0)
    save_image(figures / "full_comparison.png", straightness_panel(straightness, names))
    save_image(figures / "local_comparison.png", residual_panel(residuals, comparison_images))
    write_csv(output_dir / "line_residuals.csv", [
        {"series": label, "image_name": COMPARISON_IMAGE, "direction": direction,
         "line_index": 0, "corner_index": index, "signed_distance_px": value}
        for (label, direction), values in residuals.items()
        for index, value in enumerate(values, start=1)
    ])
    write_csv(output_dir / "parameters.csv", parameters)
    write_csv(output_dir / "reprojection.csv", errors)
    write_csv(output_dir / "reprojection_points.csv", point_errors)
    write_csv(output_dir / "linearity.csv", lines)
    write_csv(output_dir / "linearity_summary.csv", straightness)
    write_json(output_dir / "experiment.json", {
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "comparison_image": COMPARISON_IMAGE, "output_size": image_size,
        "output_camera_matrix": "Use the fitted K as both input and output (Chapter 4).",
        "crop": False,
        "illustrated_row": 0, "illustrated_column": 0,
        "signed_residual_convention": "Positive below the row or right of the column.",
        "residual_display": "Fit aligned with horizontal or vertical zero line; normal offsets enlarged equally.",
        "subset_model": SUBSET_MODEL,
        "subset_flags": BASE_FLAGS, "subset_free_distortion": ["k1", "k2"],
        "subset_square_mm": square_mm,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
