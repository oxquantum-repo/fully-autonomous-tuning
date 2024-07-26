from typing import Tuple

import cv2 as cv
import numpy as np

import bias_triangle_detection.btriangle_detection as btriangle_detection
import bias_triangle_detection.btriangle_properties as btriangle_properties
from bias_triangle_detection.utils.contours_and_masks import contours_to_mask
from bias_triangle_detection.utils.types import Contour, Mask

# centroid = (x, y), width = w, height = h, angle = theta
RotatedRectangle = Tuple[Tuple[float, float], Tuple[float, float], float]


def clip_to_contour(line: np.ndarray, contour: Contour) -> np.ndarray:
    """Intersection of line and contour. So far used for intersecting excited state line with triangle.

    Args:
        line (np.ndarray): line as start and end point.
        contour (Contour): contour to clip to.

    Returns:
        np.ndarray: array of points in the line segment intersection.
    """
    shape = tuple(np.max(contour.squeeze(), axis=0).astype(int) + 1)[::-1]
    canvas = np.zeros(shape)
    color = 255

    img_ex_line = cv.drawContours(
        canvas.copy(),
        [line.round().astype(int).reshape(-1, 1, 2)],
        -1,
        color,
        thickness=1,
    )
    image_triangle = canvas.copy()
    cv.fillPoly(image_triangle, [contour.reshape(-1, 2)], color=color)
    intersection = np.uint8(np.logical_and(img_ex_line, image_triangle) * 1)
    intersection_points, _ = cv.findContours(
        intersection, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    return intersection_points[0].reshape(-1, 2)


def get_rectangle_features(
    pulsed: np.ndarray,
    *,
    unpulsed: np.ndarray,
    res: int,
    min_area: float,
    direction: str,
    invert: int,
    thr_method: str = "triangle",
    prior_dir: str = "up",
    slope_tol: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, Mask, Mask]:
    """Get excited line segment and middle of base. Also the masks of these two.

    Args:
        pulsed (np.ndarray):
        unpulsed (np.ndarray):
        res (int):
        min_area (float):
        direction (str):
        invert (int):
        thr_method (str, optional):. Defaults to "triangle".
        prior_dir (str, optional):. Defaults to "up".
        slope_tol (float, optional):. Defaults to 0.4.

    Returns:
        Tuple[np.ndarray, np.ndarray, Mask, Mask]:
    """
    unpulsed *= invert
    pulsed *= invert

    gray_unpulsed, _, mask_unpulsed = btriangle_detection.triangle_segmentation_alg(
        unpulsed, res=res, min_area=min_area, thr_method=thr_method
    )
    base_unpulsed, corner_pts, _ = btriangle_properties.detect_base_alt_slope(
        gray_unpulsed, mask_unpulsed, direction
    )
    ex_line_unpulsed = btriangle_properties.get_excited_state_line(
        gray_unpulsed, base_unpulsed, direction, slope_tol
    )

    gray_pulsed, _, _ = btriangle_detection.triangle_segmentation_alg(
        pulsed, res=res, min_area=min_area, thr_method=thr_method
    )
    (
        _,
        _,
        _,
        matched_contour,
        translation_from_unpulsed_to_pulsed,
    ) = btriangle_detection.triangle_shape_matching(
        gray_unpulsed, gray_pulsed, mask_unpulsed, prior_dir=prior_dir
    )
    if len(ex_line_unpulsed) == 0:
        axes_points, axes, line_img = btriangle_properties.detect_detuning_axis(
            gray_unpulsed, base_unpulsed, corner_pts
        )
        ex_line_unpulsed = np.mean(axes_points, axis=1)
        excited_line = ex_line_unpulsed.reshape(-1, 2) + translation_from_unpulsed_to_pulsed
    else:
        excited_line = (
                ex_line_unpulsed[-4:].reshape(-1, 2) + translation_from_unpulsed_to_pulsed
        )
    excited_line = expand_line(excited_line).round().astype(int)
    base = base_unpulsed + translation_from_unpulsed_to_pulsed
    clipped_excited_line = clip_to_contour(excited_line, matched_contour)
    base_mean = base.mean(0)
    base = base

    min_point_index = np.argmin(clipped_excited_line[:, 0])
    max_point_index = np.argmax(clipped_excited_line[:, 0])
    end_points_excited_line = np.array([clipped_excited_line[min_point_index],
                                        clipped_excited_line[max_point_index]])

    len_of_excited_line = np.linalg.norm(end_points_excited_line[0] - end_points_excited_line[1])
    diff_vec_base = base[0] - base[1]
    diff_vec_base = diff_vec_base / np.linalg.norm(diff_vec_base)
    base_clipped = np.array([
        base_mean - diff_vec_base * len_of_excited_line / 2,
        base_mean + diff_vec_base * len_of_excited_line / 2
    ])
    clipped_excited_line = clipped_excited_line // res
    base_clipped = base_clipped.round().astype(int) // res

    return (
        clipped_excited_line,
        base_clipped,
        contours_to_mask([clipped_excited_line.reshape(-1, 1, 2)], pulsed.shape),
        contours_to_mask([base_clipped.reshape(-1, 1, 2)], pulsed.shape),
    )


def expand_line(line: np.ndarray, n_times: int = 4) -> np.ndarray:
    """Expand given line n_times on each direction.

    Args:
        line (np.ndarray): line given by start and end point.
        n_times (int, optional): number of times to expand line on each direction. Defaults to 4.

    Returns:
        np.ndarray: start and end point of expanded line.
    """
    translation = np.zeros_like(line)
    translation[0] = np.diff(line, axis=0)
    translation[1] = -translation[0]
    return line + translation * n_times
