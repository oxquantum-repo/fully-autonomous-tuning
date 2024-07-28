from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_minimum

import bias_triangle_detection.btriangle_properties as btriangle_properties

from .im_utils import img_res_gray, iou, threshold_mad


def triangle_segmentation_alg(
    img: np.ndarray,
    res: int,
    min_area: float,
    thr_method: str = "triangle",
    max_ratio: float = 0.9,
    eps: float = 0.01,
    min_vertices: int = 6,
    min_overlap: float = 0.85,
    inv: bool = False,
    denoising: bool = False,
    allow_MET: bool = False,
    direction: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Initial thresholding & contour detection of relevant areas in the image, with subsequent call to polygon simplification via the relaxed Ramer-Douglas-Peucker (RDP) algorithm per contour.
    
    Args:
    img: Original Image
    res: factor of resolution
    min_area: minimum area of contour to be detected (to avoid outliers)
    thr_method: Thresholding Method for contour detection, by default uses 'triangle'thresholding
    max_ratio: maximum ratio of total image permissible for contour 
    
    eps: Initial approximation precision provided to RDP
    min_vertices: minimum number of edges for approximated shape
    min_overlap: minimum area overlap (as per IoU) between initial & approximated contour
    inv: inverts image if shape has lower intensity compared to the background, default is False
    denoising: apply optional Gaussian denoising to smoothen segmentation, default = False
    allow_MET: False by default, if true facilitates enclosing triangle shape approximation for disconnected contours
    direction: Optional argument to be used in conjunction with allow_MET = True, specified direction of triangle

    
    
    Outputs: The grayscale image, segmented image & annotation mask 
    
    
    """

    gray = img_res_gray(img, res)
    if inv == True:
        gray = 255 - gray
    
    if denoising == True:
       sigma = 2
       gray = gaussian_filter(gray, sigma)
        
    gray_orig = np.copy(gray)    

    if thr_method == "triangle":
        _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
        
    elif thr_method == "noisy_triangle":
        clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(2, 2)) #further tunable
        gray = clahe.apply(gray)
        _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
        
    elif thr_method == "noisy_binary":
      _, threshold = cv.threshold(gray, int(np.mean(gray)), 255, cv.THRESH_BINARY)    

    elif thr_method == "otsu":
        _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)

    elif thr_method == "mad":
        k = 3
        thresh = threshold_mad(gray, k=k)
        _, threshold = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)

    elif thr_method == "minimum":
        thresh = threshold_minimum(gray)
        _, threshold = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    area_array = []
    cnt_array = []

    height, width = gray.shape[0], gray.shape[1]
    total_area = height * width

    for i, c in enumerate(contours):

        area = cv.contourArea(c)

        if area < min_area:

            continue

        if area > max_ratio * total_area:

            continue

        area_array.append(area)
        cnt_array.append(c)

    sorteddata = sorted(zip(area_array, cnt_array), key=lambda x: x[0], reverse=True)
    contours = sorted(cnt_array, key=cv.contourArea, reverse=True)

    ims = []
    masks = []

    for i in range(len(contours)):

        seg_result, mask, _ = relaxed_rdp(
            gray_orig, contours[i], eps, min_vertices, min_overlap
        )
        ims.append(seg_result)
        masks.append(mask)
    if masks:
        pred_mask = sum(masks).clip(0, 1)
    else:
        pred_mask = np.zeros_like(gray_orig)

    if allow_MET == True:
        mask_contour, _ = cv.findContours(
            pred_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        if len(mask_contour) > 1:
            pred_mask, ims = MET_approximation(gray_orig, pred_mask, direction)

    return gray_orig, ims, pred_mask    

    


def relaxed_rdp(
    img: np.ndarray,
    contour: np.ndarray,
    eps: float = 0.01,
    min_vertices: int = 6,
    min_overlap: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """By providing a starting precision to the contour approximation, this relaxation of the Ramer-Douglas-Peucker (RDP) facilitates a reduction in the number of contour edges, while maintaining sufficient overlap in area (as measured via the IoU), thereby avoiding a forceful fit to distorted shapes encompassing i.a. overlapping triangles.
    Here, given an initial approximation precision for the shape, the former is iteratively increased to facilitate a reduction in the polygon edges.
    
    Args:
    img: Original Image
    contour: detected approximate contour
    eps: approximation precision for polygon simplification
    min_vertices: stopping criterion for polygon approximation. Default value is 6, since an approximated pair of triangles will usually have 4-5.
    min_overlap: minimum Intersection over Union (IoU) value between original contour & approximation. Default is 0.85 to ensure sufficient overlap & facilitate vertex minimization.
    
    Outputs: The segmented gray-scale image, segmentation mask & determined end-precision for shape 
    
    """

    gray_orig = img.copy()
    no_vertices = 100
    overlap = 1

    while no_vertices > min_vertices and overlap > min_overlap:

        gray = gray_orig.copy()
        mask1 = np.zeros(gray.shape, np.uint8)
        cv.drawContours(mask1, [contour], -1, (255, 255, 255), -1)
        gt = mask1.clip(0, 1)

        gray = gray_orig.copy()
        mask2 = np.zeros(gray.shape, np.uint8)
        approx = cv.approxPolyDP(contour, eps * cv.arcLength(contour, True), True)
        cv.drawContours(mask2, [approx], -1, (255, 255, 255), -1)
        pred = mask2.clip(0, 1)

        eps = eps + 0.001

        no_vertices = len(approx)

        overlap = iou(gt, pred)

        cv.drawContours(gray, [approx], -1, (255, 255, 255), thickness=1)

    return gray, pred, eps

def triangle_feat_extr(
    img: np.ndarray, blocksize: int = 2, ksize: int = 3, k: float = 0.04
) -> np.ndarray:
    """Extract edge map of grayscale image, following intensity thresholding.
    blocksize, ksize & k: parameters of the corner harris detector, respectively referring to the neighborhood size, aperture parameter of the Sobel derivative & Harris
    detector free parameter. By default fixed.
    """

    ret, threshold = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    dst = cv.cornerHarris(threshold, blocksize, ksize, k)

    return dst


def scale_contour(cnt: np.ndarray, scale: int) -> np.ndarray:
    """Re-scale contour according to supplied scale factor."""
    M = cv.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def triangle_shape_matching(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    scaling: int = 1,
    prior_box: Optional[tuple[int, ...]] = None,
    prior_dir: Optional[str] = None,
    tol: int = 10,
) -> Tuple[np.ndarray, List, tuple[int, ...], np.ndarray, np.ndarray]:
    """Under the assumption that the bias triangle retains its basic contour shape, we use a template-based apporach
    & perform shape matching based on the detected edge maps of two images. The latter are employed to establish basic
    location in light of degrading image quality, while the originally detected contour is eventually mapped (subject to
    appropriate shifting, and if required, scaling) to the new location.

    Inputs:

    img1: Original grayscale image containing model triangle contour
    img2: Grayscale image to perform shape matching
    mask1: Binary mask for model triagle of img1
    scaling (optional): the scaling factor of the contour, by default 1
    prior_box: If prior knowledge on the location of the triangle exists, it can be provided in form of a bounding box, with format (x,y,c,w)
    prior_dir: Alternatively, if the bounding box cannot be supplied, one may specify the rough location of the shape in the image, options: right, left, up, down
    tol: Optional, deviation in pixels from bounding box in prior_dir

    Outputs:

    image: Annotated bounding box & matched contour in img2
    bbox: Bounding box of shape in img1
    detected_box: Detected bounding box of shape-matched edge map in img2
    new_contour: translated contour in img2
    shift: shift coords of contour

    """

    dst = triangle_feat_extr(img1)
    dst2 = np.uint8(np.zeros_like(dst))
    dst2[dst != 0] = 255

    dst3 = triangle_feat_extr(img2)
    dst4 = np.uint8(np.zeros_like(dst3))
    dst4[dst3 != 0] = 255

    contours, _ = cv.findContours(mask1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contour = contours[0]

    x, y, c, w = cv.boundingRect(contour)
    bbox = (x, y, c, w)

    template = dst2[y : y + w, :]
    template = template[:, x : x + c]

    if prior_box is not None:
        x1, y1, c1, w1 = prior_box
        dst4 = dst4[y1 : y1 + w1, :]
        dst4 = dst4[:, x1 : x1 + c1]

    if prior_dir == "left":

        x1, y1, c1, w1 = 0, 0, dst4.shape[1] // 2 + tol, dst4.shape[0]
        dst4 = dst4[y1:w1, :]
        dst4 = dst4[:, x1:c1]

    elif prior_dir == "right":
        x1, y1, c1, w1 = dst4.shape[1] // 2 - tol, 0, dst4.shape[1], dst4.shape[0]
        dst4 = dst4[y1:w1, :]
        dst4 = dst4[:, x1:c1]

    elif prior_dir == "up":
        x1, y1, c1, w1 = 0, 0, dst4.shape[1], dst4.shape[0] // 2 + tol
        dst4 = dst4[y1:w1, :]
        dst4 = dst4[:, x1:c1]

    elif prior_dir == "down":

        x1, y1, c1, w1 = 0, dst4.shape[0] // 2 - tol, dst4.shape[1], dst4.shape[0]
        dst4 = dst4[y1:w1, :]
        dst4 = dst4[:, x1:c1]

    result = cv.matchTemplate(dst4, template, cv.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)

    (startX, startY) = maxLoc

    if prior_box is not None or prior_dir is not None:
        (startX, startY) = (startX + x1, startY + y1)

    endX = startX + template.shape[1]
    endY = startY + template.shape[0]

    detected_box = []
    detected_box.append((startX, startY))
    detected_box.append((endX, endY))

    image = img2.copy()

    contour = scale_contour(contour, scaling)
    shift = [startX - x, startY - y]
    new_contour = contour + shift

    cv.rectangle(image, (startX, startY), (endX, endY), (150, 150, 0), 2)
    cv.drawContours(image, [new_contour], -1, (0, 50, 150), thickness=1)

    return image, detected_box, bbox, new_contour, shift


def MET_approximation(
    img: np.ndarray, masks: np.ndarray, direction: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the detected mask consists of multiple non-intersecting contours, the bias triangle likely features extreme gaps
    and/or distinct lines that degrade further processing. Therefore we resort to employing the Minimum Enclosing Triangle (MET)
    with respect to detected corner points in order to obtain a closed shape.

    Inputs:
    img: Original grayscale image
    masks: Originally detected mask
    direction: user-provided direction of bias triangles

    Outputs:
    New MET mask of detected shape & segmented image
    """
    img_tr = img.copy()

    base, corner_pts, c_im = btriangle_properties.detect_base(img, masks, direction)

    _, triangle = cv.minEnclosingTriangle(np.array(corner_pts))
    pts = np.int32(np.squeeze(np.round(triangle)))

    mask_new = np.zeros(img.shape, np.uint8)
    cv.drawContours(mask_new, [np.array([pts])], -1, (255, 255, 255), -1)
    pred = mask_new.clip(0, 1)

    seg_new = img.copy()
    cv.drawContours(seg_new, [np.array([pts])], -1, (255, 255, 255), 1)

    return pred, [seg_new]

def PSB_detector(
    img1: np.ndarray,
    img2: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    direction: str,
    slope_tol: float = 0.4,
    int_tol: float = 0.05,
    median: bool = False,
) -> Tuple[tuple, bool]:
    """DRAFT
    Detects Pauli Spin Blockade (True or False) by comparing the average intensity of the upper triangle segment between a blocked &
    unblocked bias triangle pair, whereby segmentation is modelled on the latter

    Inputs:
    img1: Unblocked
    img2: Blocked
    base: Base line of shape in img1
    mask: Pixelwise shape mask of img1
    direction: Direction of bias triangles
    tol: Tolerance for deviation in absolute value between slopes of detected lines
    slope_tol: Tolerance for deviation in absolute value between slopes of detected lines
    int_tol: Tolerance for PSB metric (absolute value difference between normalized segment intensities)
    median: if True, selects the median of detected lines (ordered by y-intercept), False by default, so that the line with largest 
    y-intercept (outmost) is selected
    seg_tol: Optional arg, if provided gves percentage of image length as threshold for segments that are too small

    Outputs:
    Binary (PSB) and normalized intensity values from both triangle segments
    """
    
    if len(base)==0:

        PSB = False
        intensity_pair = []

        return intensity_pair, PSB
        
    exc_state, line_mask = btriangle_properties.get_excited_state(
        img1, base, direction, slope_tol
    )

    if len(exc_state) == 0:

        PSB = False
        intensity_pair = []

        return intensity_pair, PSB
    
    
    
    idx = np.argsort(exc_state[:, 1])
    exc_state = exc_state[idx, :]
    
    if median == True:
        
        median_line = len(idx) // 2
        ex_line = exc_state[median_line]
        
    else:
        ex_line = exc_state[-1]
    
    points = ex_line[2:]

    x_coords, y_coords = (points[0], points[2]), (points[1], points[3])
    slope_b, b_intc = btriangle_properties.get_line(base.flatten())
    y_int = np.mean([points[1] - points[0]*slope_b, points[3] - points[2]*slope_b])
    n_y1 = slope_b* points[0]+y_int
    n_y2 = slope_b* points[2]+y_int
 
    ex_line = np.array([slope_b, y_int, points[0], n_y1, points[2], n_y2])

    if seg_tol is not None and abs(y_int-b_intc)<seg_tol*img1.shape[0]:

        print('Segment too small, no PSB.')
        PSB = False 
        intensity_pair = []

        return intensity_pair, PSB

    contour, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    intersection = btriangle_properties.extract_triangle_seg(
        img1, base, ex_line, contour[0]
    )
    
    if len(intersection) == 0:

        PSB = False
        intensity_pair = []

        return intensity_pair, PSB

    _, _, tr1_int_seg = btriangle_properties.extract_props(intersection, img1)

    _, _, tr1_int = btriangle_properties.extract_props(contour[0], img1)

    intersection = btriangle_properties.extract_triangle_seg(
        img2, base, ex_line, contour[0]
    )
    
    if len(intersection) == 0:

        PSB = False
        intensity_pair = []

        return intensity_pair, PSB

    _, _, tr2_int_seg = btriangle_properties.extract_props(intersection, img2)

    _, _, tr2_int = btriangle_properties.extract_props(contour[0], img2)

    intensity_pair = (tr1_int_seg / tr1_int, tr2_int_seg / tr2_int)

    if abs(tr1_int_seg / tr1_int - tr2_int_seg / tr2_int) > int_tol:

        PSB = True

    else:

        PSB = False

    return intensity_pair, PSB

