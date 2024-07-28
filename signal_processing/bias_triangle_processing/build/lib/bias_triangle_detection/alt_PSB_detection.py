from typing import List, Tuple, Optional
from bias_triangle_detection.btriangle_properties import get_line
import bias_triangle_detection.btriangle_properties as btriangle_properties
import numpy as np
import cv2 as cv
from numpy import ones, vstack, hstack
from numpy.linalg import lstsq
import math
from scipy.spatial import distance



def fit_slope_to_base(points: np.ndarray, slope_b: float, b_intc: float, idx: int) -> np.ndarray:
    """ Given two end points of a line, fit a line parallel to the base between them. This is to ensure potential 
    excited state lines remain parallel to the base.
    
    """

    x_coords, y_coords = (points[0], points[2]), (points[1], points[3])
    
    y_int = np.mean([points[1] - points[0]*slope_b, points[3] - points[2]*slope_b])
    
    n_y1 = slope_b* points[0] + y_int
    n_y2 = slope_b* points[2] + y_int
    
    new_line = np.array([slope_b, y_int, points[0], n_y1, points[2], n_y2, idx])
    
    return new_line


def line_detector_new (lines: np.ndarray, tol: int, slope_b: float, b_intc: float) -> List[float]:
    """Given detected lines, filter out those within a tolerance range to the base line slope & apply
    fitting with respect to the latter.
    """

    line_list = []
    
    for idx, line in enumerate(lines):

        slope, intc = get_line(line.flatten())

        if np.abs(slope_b - slope) < tol:
            
            new_line = fit_slope_to_base(line.flatten(), slope_b, b_intc, idx)
            
            line_list.append([new_line])

    return line_list


def get_excited_state_alt(
    img: np.ndarray, base: np.ndarray, direction: str, tol: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Given the base & direction of the detected triangle, detect & characterize remaining (parallel) lines to further deduce location of the 1st excited state

    Args:
    img: Original Image
    base: detected base of triangle
    direction: direction of the triangle
    tol: Tolerance for devation from base line slope

    Outputs:
    excited states (incl. slope, intercept & end points)
    image with detected lines

    """

    slope_b, b_intc = get_line(base.flatten())

    lines = cv.ximgproc.createFastLineDetector().detect(img)

    if lines is None:

        print("No lines found")

        return [], []

    line_list = line_detector_new(lines, tol, slope_b, b_intc)

    if len(line_list) == 0:

        tol = tol + 0.1
        line_list = line_detector_new(lines, tol, slope_b, b_intc)

    if len(line_list) == 0:

        print("No lines found, potentially increase tolerance for slope deviation.")

        return [], []

    l_list = np.array(line_list)
    l_list = l_list.reshape(l_list.shape[0],l_list.shape[2])

    if direction == "up" or direction == "right":

        try:
            exc_state = l_list[l_list[:, 1] < b_intc]
        except:
            exc_state = l_list

    elif direction == "down" or direction == "left":

        try:
            exc_state = l_list[l_list[:, 1] > b_intc]
        except:
            exc_state = l_list

    mask = img.copy()

    new_id = list(map(int, exc_state[:, -1]))
    state_lns = lines[new_id]
    

    for ln in state_lns:
        x1 = int(ln[0][0])
        y1 = int(ln[0][1])
        x2 = int(ln[0][2])
        y2 = int(ln[0][3])

        cv.line(mask, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

    
    exc_states = exc_state[:, :6]
    
    return exc_states, mask

def find_longest_side(img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Alternative routine for baseline extraction - recommended when triangles are not centered (likely to appear only partially in scan). Following basic polygon approximation & vertex reduction, the longest side is computed.
    Inputs:
    img: Original grayscale image
    masks: Segmentation mask
    
    Outputs:
    lside: end points of base
    img_b: Baseline in image
    
    """
    
    img_b = img.copy()
    c, _ = cv.findContours(masks,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    peri = cv.arcLength(c[0], True)
    approx = cv.approxPolyDP(c[0], 0.01 * peri, True)
    approx = approx.reshape(approx.shape[0],approx.shape[2])
    
    
    d = np.diff(np.vstack([approx,approx[0]]), axis=0)
    
    segdists = np.hypot(d[:,0], d[:,1])
    
    idx1, idx2 = segdists.argmax(), (segdists.argmax()+1)%len(approx)
    
    
    
    lside = approx[[idx1,idx2],:]
    
    cv.polylines(img_b, np.array([lside]), True, [200, 200, 0], 1)
    
    return lside, img_b



def PSB_detector_new(
    img1: np.ndarray,
    img2: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    direction: str,
    slope_tol: float = 0.4,
    int_tol: float = 0.05,
    median: bool = False,
    seg_tol: Optional[float] = None,
) -> Tuple[tuple, bool]:
    """DRAFT 2
    Detects Pauli Spin Blockade (True or False) by comparing the average intensity of the upper triangle segment between a blocked &
    unblocked bias triangle pair, whereby segmentation is modelled on the latter

    Inputs:
    img1: Unblocked
    img2: Blocked
    base: Base line of shape in img1
    mask: Pixelwise shape mask of img1
    direction: Direction of bias triangles
    slope_tol: Tolerance for deviation in absolute value between slopes of detected lines
    int_tol: Tolerance for PSB metric (absolute value difference between normalized segment intensities)
    median: if True, selects the median of detected lines (ordered by distance), False by default, so that the line with largest 
    distance (outmost) is selected
    seg_tol: Optional arg, if provided gves percentage of image length as threshold for segments that are too small

    Outputs:
    Binary (PSB) and normalized intensity values from both triangle segments
    """


   
    if len(base)==0:

        PSB = False
        intensity_pair = []

        return intensity_pair, PSB

    exc_state, line_mask = get_excited_state_alt(
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

def get_triangle_seg(img1: np.ndarray,img2: np.ndarray, base: np.ndarray,mask: np.ndarray,direction: str,shift: np.ndarray, slope_tol: float = 0.4) -> Tuple[np.ndarray, np.ndarray]: 
    """
    img1: pulsed image
    img2: original unpulsed image
    """

    if len(img2) == 0:
        img2 = img1

    
    ex_line = btriangle_properties.get_excited_state_line(img2, base, direction, slope_tol)
    if len(ex_line)==0:
        print('No lines detected')
        return [], []
    
    ex_line[2:] = ex_line[2:]+shift*2
    
    contour, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    intersection = btriangle_properties.extract_triangle_seg(
        img1, base+shift, ex_line, contour[0]
    )
    
    if len(intersection)==0:
        print('Likely no lines detected.')
        return [], []
    
    blank = img1.copy()
    seg_im = cv.drawContours(blank, [intersection], -1,(255,0,255), thickness=1)
    
    return intersection, seg_im    


     
