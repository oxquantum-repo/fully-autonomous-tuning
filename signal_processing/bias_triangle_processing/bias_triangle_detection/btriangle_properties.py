from typing import Tuple, List, Optional
import numpy as np
import cv2 as cv
from numpy import ones, vstack, hstack
from numpy.linalg import lstsq
import math
from scipy.spatial import distance
from functools import reduce
import operator



def get_line(points: np.ndarray) -> Tuple[float, float]:
    """
    Compute slope and intercept from given set of points.
    
    """
    x_coords, y_coords = (points[0], points[2]), (points[1], points[3])
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    return m, c

def midpoint(p1: tuple[int, int], p2: tuple[int, int]) -> Tuple[int, int]:
    """
    Compute midpoint between two points.
    """
    return int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)  

def line_detector(lines: np.ndarray, tol: float, slope_b: float) -> List[float]:

    line_list = []
    for idx, line in enumerate(lines):

        slope, intc = get_line(line.flatten())

        if np.abs(slope_b - slope) < tol:
            line_list.append([slope, intc, idx])

    return line_list


def get_excited_state(
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

    line_list = line_detector(lines, tol, slope_b)

    if len(line_list) == 0:

        tol = tol + 0.1
        line_list = line_detector(lines, tol, slope_b)

    if len(line_list) == 0:

        print("No lines found, potentially increase tolerance for slope deviation.")

        return [], []
    
    l_list = np.array(line_list)

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

    new_id = list(map(int, exc_state[:, 2]))
    state_lns = lines[new_id]

    for ln in state_lns:
        x1 = int(ln[0][0])
        y1 = int(ln[0][1])
        x2 = int(ln[0][2])
        y2 = int(ln[0][3])

        cv.line(mask, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

    exc_states = hstack(
        [exc_state[:, :2], state_lns.reshape(state_lns.shape[0], state_lns.shape[2])]
    )

    return exc_states, mask


def extract_triangle_seg(
    img: np.ndarray, base: np.ndarray, ex_line: np.ndarray, contour: np.ndarray
) -> np.ndarray:
    """ Given the base line and another line of interest within the triangle (i.e. potential excited state), compute the segment enclosed by the two lines. 
    
    Args: 
    img: Original image
    base: baseline of triangle
    ex_line: line within the triangle (for excited state)
    contour: previously detected triangle contour
    
    Outputs: 
    The triangle segment
    
    """

    p1, p2 = base
    q1, q2 = np.int32(ex_line[2:].reshape(2, 2))

    blank = np.zeros(img.shape[0:2])
    line_img = np.zeros(img.shape[0:2])

    h, w = line_img.shape[:2]
    x1 = 0
    m1, _ = get_line(base.flatten())
    y1 = -(p1[0] - 0) * m1 + p1[1]
    x2 = w
    y2 = -(p2[0] - w) * m1 + p2[1]

    x3 = 0
    m2 = ex_line[0]
    y3 = -(q1[0] - 0) * m2 + q1[1]
    x4 = w
    y4 = -(q2[0] - w) * m2 + q2[1]

    pts = np.array(
        [(int(x3), int(y3)), (int(x4), int(y4)), (int(x2), int(y2)), (int(x1), int(y1))]
    )
    cv.fillPoly(line_img, [pts], color=(255, 0, 0))

    image2 = cv.drawContours(blank.copy(), [contour], -1, (255, 0, 255), thickness=-1)

    intersection = np.uint8(np.logical_and(line_img, image2) * 1)
    int_cnt, _ = cv.findContours(intersection, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(int_cnt) < 1:
        return []

    return int_cnt[0]


def extract_props(
    contour: np.ndarray, img: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    """ Compute the area, centroid & mean intensity of a contour.
    
    Args:
    contour: Segmented contour
    img: Original image
    
    
    Outputs:
    area
    position
    intensity
    """

    c = contour
    M = cv.moments(c)

    area = M["m00"]

    if M["m00"] == 0.0:

        position = np.array([0, 0])

    else:

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        position = np.array([cX, cY])

    perimeter = cv.arcLength(c, True)

    img_new = np.zeros(img.shape, np.uint8)
    cv.drawContours(img_new, [c], -1, (255, 255, 255), -1)
    mask_contour = img_new == 255
    intensity = np.mean(img[mask_contour])

    return area, position, intensity


def compute_angles(
    img: np.ndarray, contour: np.ndarray, approx_type: str
) -> Tuple[List, np.ndarray]:

    """Given an array of points & specification of triangle approximation method:
    Minimum Enclosing Triangle (MET) or Minimum Polygon Approximation via RDP
    compute the angles in the triangle
    MET: In the case of two merging triangles, the MET will enclose the extreme points
    RDP: To be chosen over MET, if the shape to be approximated is closer to one rather than two merging triangles
    
    
    Args:
    img: Original image
    contour: Segmented contour
    approx_type: triangle approximation method (MET or RDP)
    
    Outputs:
    angles
    image with annotated triangle used
    """

    if approx_type == "MET":

        img_tr = img.copy()
        _, triangle = cv.minEnclosingTriangle(contour)
        pts = np.int32(np.squeeze(np.round(triangle)))
        cv.polylines(img_tr, np.array([pts]), True, [255, 0, 255], 1)

    elif approx_type == "RDP":

        img_tr = img.copy()
        eps = 0.01

        while approx.shape[0] > 3:

            approx = cv.approxPolyDP(contour, eps * cv.arcLength(contour, True), True)
            eps = eps + 0.01

        cv.polylines(img_tr, [approx], True, [255, 0, 255], 1)
        pts = approx.reshape(approx.shape[0], approx.shape[2])

    l1 = distance.euclidean(tuple(pts[0]), tuple(pts[2]))
    l2 = distance.euclidean(tuple(pts[1]), tuple(pts[2]))
    l3 = distance.euclidean(tuple(pts[0]), tuple(pts[1]))

    A1 = np.degrees(np.arccos((l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)))
    A2 = np.degrees(np.arccos((l3 ** 2 + l2 ** 2 - l1 ** 2) / (2 * l3 * l2)))
    A3 = np.degrees(np.arccos((l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)))

    return list([A1, A2, A3]), img_tr


def detect_base(
    img: np.ndarray, masks: np.ndarray, direction: str
) -> Tuple[np.ndarray, List, np.ndarray]:
    """Given the segmented shape mask & a user-specified direction of the shape, compute the extreme corner points & determine the base line of the triangle
    
    Args:
    gray_orig: Original grayscale image
    masks: Segmentation of detected triangle
    direction: user-specified direction of triangle
    
    Outputs: 
    base: the base points, 
    points: all corner points, 
    img_b: the image with the annotated features
   
    """

    cnt = cv.findNonZero(masks)
    img_b = img.copy()

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    points = list(set([leftmost, rightmost, topmost, bottommost]))

    for point in points:
        cv.circle(img_b, point, 1, (155, 255, 0), -1)

    if direction == "left":
        direct = leftmost
    elif direction == "right":
        direct = rightmost
    elif direction == "up":
        direct = topmost
    elif direction == "down":
        direct = bottommost

    else:
        raise ValueError(
            "Please provide a valid direction: "
            "up"
            ", "
            "down"
            ", "
            "right"
            " or "
            "left"
            ". "
        )

    distances = np.linalg.norm(np.array(points) - np.array(direct), axis=1)
    nonzeros = np.where(distances)[0]
    idx = np.argsort((-1) * distances[nonzeros])
    
    base = np.array(points)[nonzeros[idx[:2]]]

    cv.polylines(img_b, np.array([base]), True, [100, 100, 0], 1)

    return base, points, img_b

def detect_detuning_axis(gray: np.ndarray, base: np.ndarray, corner_points: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
    """ Extract the detuning axis, i.e. the line through the triangle tip perpendicular to the base. 
    This is done for each triangle pair.
    
    Inputs:
    gray: Original image
    base: detected base line
    corner_points: extreme corner points
    
    Ouputs:
    The axes' end points, line specifications and their annotations on the image.
    
    """
    
    base_l = sorted(list(map(tuple, base))) 
    tips = sorted(list(set(corner_points).difference(set(base_l))))
    
    if len(tips) == 1:

        mid = midpoint(*base_l)
        line_img = gray.copy()

        m1, c1 = get_line(np.array([tips[0], mid]).flatten())

        axes_specs = [(m1, c1)]
        axes_points = [(tips[0], mid)]
        cv.line(line_img, pt1=mid, pt2=tips[0], color=(150, 150, 0), thickness=1)
        print('Warning: Only one detuning axis detected, consider centering.')
        
        return axes_points, axes_specs, line_img


    
    mid = midpoint(*base_l)
    mid1 = midpoint(base_l[0], mid)
    mid2 = midpoint(base_l[1], mid)
    
    line_img = gray.copy()
    
    try:

      m1, c1 = get_line(np.array([tips[0], mid1]).flatten())

      m2, c2 = get_line(np.array([tips[1], mid2]).flatten())

      axes_specs = [(m1, c1), (m2, c2)]
      axes_points = [(tips[0], mid1), (tips[1], mid2)]

      cv.line(line_img, pt1=mid1, pt2=tips[0], color=(150, 150, 0), thickness=1)
      cv.line(line_img, pt1=mid2, pt2=tips[1], color=(150, 150, 0), thickness=1)

    except:
        
        print('No two detuning axes were detected. Retake measurement to center triangle.')
        return [],[],[]
    
    
    return axes_points, axes_specs, line_img    


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

def find_detuning_axes(img: np.ndarray, mask: np.ndarray, direction: str, basis_cor: bool = False) -> Tuple[List[int], List[int], np.ndarray]:


    base, corner_pts, c_im = detect_base(img, mask, direction)
    
    if basis_cor == True:

        base, c_im = find_longest_side(img, mask)
        
    axes_points,axes,line_img = detect_detuning_axis(img, base, corner_pts)
    
    return axes_points,axes,line_img   

def detect_base_alt(img: np.ndarray, masks: np.ndarray, direction: str
) -> Tuple[np.ndarray, List, np.ndarray]:
    
    cnt = cv.findNonZero(masks)
    img_b = img.copy()

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    points = list(set([leftmost, rightmost, topmost, bottommost]))

    for point in points:
        cv.circle(img_b, point, 1, (155, 255, 0), -1)

    if direction == "left":
        direct = leftmost
    elif direction == "right":
        direct = rightmost
    elif direction == "up":
        direct = topmost
    elif direction == "down":
        direct = bottommost

    else:
        raise ValueError(
            "Please provide a valid direction: "
            "up"
            ", "
            "down"
            ", "
            "right"
            " or "
            "left"
            ". "
        )






    coords = list(map(tuple, points))
    
    

    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    corner_pts = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    corner_pts = np.array(corner_pts)
    
    d = np.diff(np.vstack([corner_pts,corner_pts[0]]), axis=0)

    tip_idx = np.where(np.all(corner_pts==direct,axis=1))[0].item()

    segdists = np.hypot(d[:,0], d[:,1])
    segdists[[tip_idx,tip_idx-1]] = 0
    
    idx1, idx2 = segdists.argmax(), (segdists.argmax()+1)%len(corner_pts)
    
    
    
    base = corner_pts[[idx1,idx2],:]
    
    cv.polylines(img_b, np.array([base]), True, [200, 200, 0], 1)

    corner_points = list(map(tuple, corner_pts))
    
    return base, corner_points, img_b    

def location_and_box_by_contour(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, List, List]:

    positions = []
    bounding_boxes = []
    contours, _ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area, position, intensity = extract_props(contour, img)
        positions.append(position)
        bounding_box = np.min(contour, axis=0).flatten().tolist() + np.max(contour, axis=0).flatten().tolist()
        bounding_boxes.append(bounding_box)

    img_new = img.copy()
    for point, bounding_box in zip(positions, bounding_boxes):
        cv.circle(img_new,tuple(point),3,(255,255,255))
        cv.rectangle(img_new, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 255, 255), 1)

    return img_new, positions, bounding_boxes

def location_by_contour(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, List]:

    positions = []
    contours, _ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
    
        area, position, intensity = extract_props(contour, img)
        positions.append(position)
    
    img_new = img.copy()
    for point in positions:
    
         cv.circle(img_new,tuple(point),3,(255,255,255))
        
    return img_new, positions  

def find_intersection(line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
    """Finds the intersection point between two lines.

    Inputs:
    line1 & line2: Two lines specified via two pairs of coordinates, respetively

    Output: point of intersection
    """

    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))


    return Px, Py   


def detect_refined_detuning_axis(img: np.ndarray, img2: np.ndarray, base: np.ndarray, corner_pts: np.ndarray, shift: np.ndarray, direction: str = 'down', slope_tol: float = 0.4, padding_factor: Optional[float] = None) -> Tuple[np.ndarray, List]:
    """Extracts shortened detuning axes.

    Inputs:
    img: Original image
    base: Previously extracted base
    corner_pts: Previously extracted corner points
    shift: Shift of template contour wrt original image
    direction: Direction of triangles
    slope_tol: Tolerance for slope deviation
    padding_factor: Optional argument to extend past the detected detuning line as a factor of its length

    Outputs:
    Coordinates of detuning axes and plot
    """

    axes_points,_ ,_  = detect_detuning_axis(img, base, corner_pts)

    if len(img2)==0:
        
        img2 = img
    
    ex_line = get_excited_state_line(img2, base-shift, direction, slope_tol)

    if len(ex_line)==0:

        print('No excited state lines detected - consider increasing slope tolerance or move on.')

        return [], []

    ex_line[2:] = ex_line[2:]+shift*2
    
    if len(ex_line) == 0:

        print('No lines detected - consider increasing slope tolerance or move on.')

        return [], []
    
    if len(axes_points)<=1:

        print('Less than two detuning axes detected. Consider centering or skip.')

        return [],[]

    else:    
    
       px1, py1 = find_intersection(np.array(axes_points[0]).flatten(), ex_line[2:])
    
       px2, py2 = find_intersection(np.array(axes_points[1]).flatten(), ex_line[2:])
    
    
       axes_points_cut = list([((int(px1), int(py1)),axes_points[0][1]), ((int(px2), int(py2)),axes_points[1][1])])
    
    
       if padding_factor is not None:
        
           dist = np.array(axes_points_cut[0][0])-np.array(axes_points_cut[0][1])
           norm_b = dist/np.linalg.norm(dist)
        
           dist = np.linalg.norm(dist)*padding_factor
        
           if direction == 'down':
               new = (axes_points_cut[0][1]+dist*norm_b).astype('int')
           else:    
               new = (axes_points_cut[0][1]-dist*norm_b).astype('int')
        
        
           dist2 = np.array(axes_points_cut[1][0])-np.array(axes_points_cut[1][1])
           norm_b2 = dist2/np.linalg.norm(dist2)
        
           dist2 = np.linalg.norm(dist2)*padding_factor
        
           if direction == 'down':
                new2 = (axes_points_cut[1][1]+dist2*norm_b2).astype('int')
           else:      
                new2 = (axes_points_cut[1][1]-dist2*norm_b2).astype('int')

        
           axes_points_cut = list([(new, axes_points[0][1]), (new2, axes_points[1][1])])
        
       line_img = img.copy()
        
        
       cv.line(line_img, pt1=axes_points_cut[0][0], pt2=axes_points_cut[0][1], color=(250, 250, 0), thickness=1)
       cv.line(line_img, pt1=axes_points_cut[1][0], pt2=axes_points_cut[1][1], color=(250, 250, 0), thickness=1)
        
    return line_img, axes_points_cut   

def get_excited_state_line(img1: np.ndarray, base: np.ndarray, direction: str, slope_tol: float) -> np.ndarray:
    """Extracts outmost excited state line of bias triangle.

    Inputs:
    img1: Original image
    base: Previously extracted base.
    direction: Direction of triangle
    slope_tol: Tolerance for slope deviation

    Outputs:
    Specifics of outmost excited state line in form of slope, y-intercept and two points.

    """
    exc_state, line_mask = get_excited_state_alt(
        img1, base, direction, slope_tol
    )

    if len(base) == 0:
        
        exc_ln =[]
        
        print('No base detected.')
        
        return exc_ln

    if len(exc_state) == 0:

        exc_ln = []
        
        return exc_ln
    
    idx = np.argsort(exc_state[:, 1])
    exc_state = exc_state[idx, :]
    ex_line = exc_state[-1]

    points = ex_line[2:]

    x_coords, y_coords = (points[0], points[2]), (points[1], points[3])
    slope_b, b_intc = get_line(base.flatten())
    y_int = np.mean([points[1] - points[0]*slope_b, points[3] - points[2]*slope_b])
    n_y1 = slope_b* points[0]+y_int
    n_y2 = slope_b* points[2]+y_int
 
    ex_ln = np.array([slope_b, y_int, points[0], n_y1, points[2], n_y2])
    
    return ex_ln 

def detect_base_alt_slope(img: np.ndarray, masks: np.ndarray, direction: str
) -> Tuple[np.ndarray, List, np.ndarray]:

    """
    Alternative baseline computation by slope. Assumes downward-left direction (hence here a positive slope).

    Args:
    gray_orig: Original grayscale image
    masks: Segmentation of detected triangle
    direction: user-specified direction of triangle

    Outputs:
    base: the base points,
    points: all corner points,
    img_b: the image with the annotated features
    """
    
    cnt = cv.findNonZero(masks)
    img_b = img.copy()

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    points = list(set([leftmost, rightmost, topmost, bottommost]))

    for point in points:
        cv.circle(img_b, point, 1, (155, 255, 0), -1)

    if direction == "left":
        direct = leftmost
    elif direction == "right":
        direct = rightmost
    elif direction == "up":
        direct = topmost
    elif direction == "down":
        direct = bottommost

    else:
        raise ValueError(
            "Please provide a valid direction: "
            "up"
            ", "
            "down"
            ", "
            "right"
            " or "
            "left"
            ". "
        )




    coords = list(map(tuple, points))
    
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    corner_pts = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    corner_pts = np.array(corner_pts)
    
    
    corner_pts_n = np.vstack([corner_pts,corner_pts[0]])
    
    slopes = []
    
    for i in range(len(corner_pts_n)-1):
    
         m, _ = get_line(corner_pts_n[i:i+2].flatten())
         slopes.append(m)
                             
    ids = np.where(np.all(corner_pts==direct,axis=1))[0].item()  
                             
    slopes[ids] = 0
    slopes[(ids-1)%len(corner_pts)] = 0
                             
    idx1 = np.argwhere(np.array(slopes)>0)
    
    if len(idx1)==0:
        
        print('No proper baseline slope detected, segmentation may have failed.')
        
        base, corner_pts, c_im = [],[],[]
        
        
        return base, corner_pts, c_im
        
    idx2 = (idx1+1)%len(corner_pts)
    
    idxs =[idx1.item(),idx2.item()]
    base = corner_pts[idxs,:]
    
    
    cv.polylines(img_b, np.array([base]), True, [200, 200, 0], 1)

    corner_points = list(map(tuple, corner_pts))
    
    return base, corner_points, img_b 

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

