from typing import Callable, Tuple, Generator, List
import numpy as np 


def evaluate_contours(starting_point: Tuple[int,int], measure: Callable[Tuple[int, int], bool], bounds: Tuple[int,int], MAX_SEARCH_DISTANCE: int=5) -> np.ndarray: 
    """Evaluates contours using Moore's Neighbor Tracing algorithm. The algorithm will stop when it reaches the 
    starting point again. A starting point might be outside of a feature. Should this occur, the algorithm will search for the 
    nearest feature point (pixel with a value = 1) in each of the cardinal directions. Contour tracing will be ran on 
    each point, meaning a maximum of 4 contours will be returned and a minium of zero (should no contours be found in the search).

    Args:
        starting_point (Tuple[int,int]): The initial point the algorithm should starting measuring from
        measure (Callable[Tuple[int, int], float]): A function to perform the measurement 

    Returns:
        np.ndarray: The contour of the local feature
    """        
    visited = np.zeros(bounds, dtype=bool) 
    measurements = np.zeros(bounds, dtype=bool) 
    contours = [] 

    # first we need to find a starting point that is at the edge of a contour 
    try:
        current_previous_point_pairs, measurements = evaluate_contour_starting_points(
            starting_point, 
            measure, 
            measurements, 
            bounds, 
            MAX_SEARCH_DISTANCE
        )
    except ValueError:
        return contours
        
    for starting_point, previous_point in current_previous_point_pairs:
        contour, visited, measurements = evaluate_contour(
            starting_point, 
            previous_point, 
            measure, 
            bounds,
            measurements, 
            visited, 
        )
        contours.append(contour)
    return contours

def evaluate_contour(starting_point: Tuple[int,int], previous_point: Tuple[int, int], measure: Callable[Tuple[int, int], bool],
    bounds: Tuple[int,int], measurements: np.ndarray, visited: np.ndarray) -> Tuple[List[Tuple[int,int]], np.ndarray, np.ndarray]: 
    """Evaluates the contour using Moore's Neighbor Tracing algorithm. The algorithm will stop when it reaches the 
    starting point again. 

    Args:
        starting_point (Tuple[int,int]): The initial point the algorithm should starting measuring from
        measure (Callable[Tuple[int, int], float]): A function to perform the measurement 

    Returns:
        np.ndarray: The contour of the local feature
    """        
    current_point = starting_point 
    assert measurements[current_point], f"The current point should be a feature point {current_point}, {previous_point} = {measurements[previous_point]}, {measurements[current_point]}"
    assert not measurements[previous_point], f"The previous point should not be a feature point {current_point}, {previous_point} = {measurements[previous_point]}, {measurements[current_point]}"
    
    contour = [] 
    
    # Start by measuring the starting point 
    visited[current_point] = True 

    MAX_MEASUREMENTS = 1500 
    measurement_count = 0
    processing = True

    while processing:
        # iterate clockwise around the current point until you find a
        # another feature point
        neighbours = clockwise_neighbours_from_point(current_point, previous_point, bounds)
        for idx, neighbour in enumerate(neighbours): 
            # Measure this point if we haven't done so already 
            if not visited[neighbour]:
                visited[neighbour] = True
                measurements[neighbour] = measure(neighbour) 
            # If this point is a feature point, add it to the contour and 
            # move the current point along 
            if measurements[neighbour]:
                contour.append(neighbour)
                if idx > 0:
                    previous_point = neighbours[idx-1]
                else:
                    previous_point = current_point
                current_point = neighbour
                break 

        measurement_count += 1 

        if measurement_count >= MAX_MEASUREMENTS:
            print(f"WARNING")
            print(f"Reached max measurements - stopping contour search.\nStart Point: {starting_point}\nCurrent Point: {current_point}\nPrevious Point: {previous_point}\nNumber of neigbours: {len(neighbours)}")
        processing = current_point != starting_point and measurement_count < MAX_MEASUREMENTS

    return contour, visited, measurements
    
def evaluate_contour_starting_points(starting_point: Tuple[int, int], measure: Callable, measurements: np.ndarray, bounds: Tuple[int,int], MAX_SEARCH_DISTANCE: int) -> Tuple[Tuple[int,int], Tuple[int,int], np.ndarray]:
    """The contour starting point needs to be on the edge of a triangle. We are handed a starting point that maybe inside 
    or outside a bias triangle. We treat these two cases separtely. For the algorithm to work, we require a measurement is not part of 
    the contour from which the starting point can be entered. 

    This method attempts to find this starting point. If the starting point is outside of a triangle, it will scan for 10 units and then
    up for 10 units halfway along that 10 unit long horizontal line. If it still does not find anything, it throws a ValueError exception.

    Args:
        starting_point (_type_): _description_
        measure (_type_): _description_
        measurements (_type_): _description_
        bounds (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Tuple[Tuple[int,int], Tuple[int,int], np.ndarray]: _description_
    """    
    current_point = starting_point
    measurements[current_point] = measure(current_point)
    current_previous_point_pairs = [] 
    if measurements[current_point]:
        # If the starting point is within the triangle, we need to find the first point outside it 
        while measurements[current_point]:
            previous_point = current_point
            current_point = (current_point[0], current_point[1]+1)
            if current_point[1] >= bounds[1]:
                raise ValueError("Starting point is invalid")
            measurements[current_point] = measure(current_point)

        tmp = previous_point
        previous_point = current_point
        current_point = tmp
        if measurements[current_point] and not measurements[previous_point] :
            current_previous_point_pairs.append((current_point, previous_point))
    else:
        # if the starting point is outside the triangle, we need to find the first point inside it
        # Draw a cross out from the starting point (which we know is a NOT a feature point)
        for velocity in [(0,1), (0,-1), (1,0), (-1,0)]:
            success = False 
            distance = 0
            current_point = starting_point 

            while not success:
                previous_point = current_point
                current_point = (current_point[0] + velocity[0], current_point[1] + velocity[1])
                distance += 1   
                if current_point[1] >= bounds[1] or distance > MAX_SEARCH_DISTANCE or current_point[0] >= bounds[0] \
                    or current_point[1] < 0 or current_point[0] < 0:
                    break 
            
                measurements[previous_point] = measure(previous_point)
                measurements[current_point] = measure(current_point)

                # if the previous point is somehow a measurement but the current point is, swap them
                if not measurements[current_point] and measurements[previous_point]:
                    current_point, previous_point = previous_point, current_point 
            
                success = measurements[current_point] and not measurements[previous_point] 

            if success:
                current_previous_point_pairs.append((current_point, previous_point))
                
    return current_previous_point_pairs, measurements
    
def clockwise_neighbours_from_point(point: Tuple[int, int], previous_point: Tuple[int,int], bounds: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
    """Evaluates the neighbours of a point within the bounds of the image by by moving to the left
    then clockwise around the point.

    Args:
        point (Tuple[int, int]): _description_
        bounds (Tuple[int, int]): _description_

    Yields:
        Generator[Tuple[int, int], None, None]: _description_
    """    
    neighbours = []

    # There's probably a more elegant way to do this...
    # it ain't pretty, but it works 
    neighbours.append((point[0], point[1]-1))
    neighbours.append((point[0]-1, point[1]-1))
    neighbours.append((point[0]-1, point[1]))
    neighbours.append((point[0]-1, point[1]+1))
    neighbours.append((point[0], point[1]+1))
    neighbours.append((point[0]+1, point[1]+1))
    neighbours.append((point[0]+1, point[1]))
    neighbours.append((point[0]+1, point[1]-1))

    valid = [is_valid(neighbour, bounds) for neighbour in neighbours] 

    # We want to reorder the neighbours so that all invalid points are removed from the list
    # but all points are still adjacent to each other. This is really only necessary when the
    # we are next to the edge of the image. 
    if not all(valid):
        first_idx, last_idx = valid.index(False), len(valid) - 1 - valid[::-1].index(False) 
        if first_idx == 0 and last_idx == len(valid) - 1:
            # The valid points are in the middle of the list, i.e. valid=[0,0,1,1,1,0,0]
            # First remove the invalid points from the start of the list 
            n_at_start = 0
            while not valid[n_at_start]:
                n_at_start += 1
            valid = valid[n_at_start:]
            neighbours = neighbours[n_at_start:]

            # Now find the invalid points at the end of the list
            first_idx, last_idx = valid.index(False), len(valid) - 1 - valid[::-1].index(False) 
        # The invalid points should now be in the middle of the list, i.e. [1,1,0,0,0,1]
        # trim those off 
        return neighbours[last_idx+1:] + neighbours[:first_idx] 


    idx = neighbours.index(previous_point)
    if idx >= 0:
        neighbours = neighbours[idx:] + neighbours[:idx]
    return neighbours

def is_valid(point: Tuple[int, int], bounds: Tuple[int, int]) -> bool:
    """Checks if a point is within the bounds of the image

    Args:
        point (Tuple[int, int]): _description_
        bounds (Tuple[int, int]): _description_

    Returns:
        bool: _description_
    """    
    return point[0] >= 0 and point[0] < bounds[0] and point[1] >= 0 and point[1] < bounds[1]