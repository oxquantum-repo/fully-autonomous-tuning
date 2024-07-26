import numpy as np 

from scipy.optimize import minimize
from typing import Dict, Tuple, List

class SkewedGrid:
    """A class that represents with an arbitrary skew and separation along each grid axis."""
    def __init__(self, initial_point: Tuple[int, int]=[0,0], axis_1: Tuple[float, float] = (0.0, 4.0), axis_2: Tuple[float,float] = (4.0,0.0) ) -> None:
        self.initial_point = initial_point
        self.axis_1 = axis_1
        self.axis_2 = axis_2 

    def fit(self, peak_locations: List[Tuple[int, int]]) -> None:
        self.initial_point, self.axis_1, self.axis_2 = self.evaluated_skewed_grid_params(peak_locations)

    def distance_to_nearest_grid_intersection(self, point: Tuple[int,int]) -> float:
        # Calculate a vector between the initinial point and the passed point
        distance = [self.initial_point[0] - point[0], self.initial_point[1] - point[1]] 

        # now change the basis of the distance vector such that it's in the basis of the grid 
        distance_in_grid_basis = np.linalg.solve(np.array([self.axis_1, self.axis_2]), distance)
        
        # Now just modulo each component by the length of the axis vectors to get it's 
        # distance from the a grid point  
        distance_in_grid_basis = [
            np.sign(distance_in_grid_basis[0]) * (np.abs(distance_in_grid_basis[0]) % 1), # modulo 1 because we have already accounted for the non-normalized basis vectors 
            np.sign(distance_in_grid_basis[1]) * (np.abs(distance_in_grid_basis[1]) % 1), 
        ]

        # if its less than 0.5 then it's closer to the next grid point along 
        if distance_in_grid_basis[0] < -0.5:
            distance_in_grid_basis[0] += 1.0
        if distance_in_grid_basis[1] < -0.5:
            distance_in_grid_basis[1] += 1.0

        # Times by the length of the basis vectors to give us distances in real grid space 
        # If we don't do this, then we will get the direction of the grid basis okay 
        # but the separation between grid points will be wrong 
        length_corrected_distance = [
            distance_in_grid_basis[0] * np.sqrt(np.sum(np.power(self.axis_1, 2.0))),
            distance_in_grid_basis[1] * np.sqrt(np.sum(np.power(self.axis_2, 2.0))),
        ]

        # Retur n the euclidean distance to the grid point 
        return  np.sqrt(np.sum(np.power(length_corrected_distance, 2.0)))

    def __repr__(self) -> str:
        return f"SkewedGrid({self.initial_point}, {self.axis_1}, {self.axis_2})" 

    def evaluated_skewed_grid_params(self, peak_locations: List[Tuple[int, int]]) -> Tuple[Tuple,...]:
        """Given a set of peaks from a number of line scans across an image with repeated structures, evaluate
        evaluate a skewed and rotated grid that connects those structures. 

        Args:
            structure_locations (RepeatedStructureLocations): The structure locations to evaluate

        Returns:
            initial_point: An initial point that is an intersection on the skewed grid 
            axis_1: A 2D vector representing the first basis of the grid
            axis_2: A 2D vector representing the second basis of the grid
        """

        def _summed_distance_to_grid(grid_params: Tuple[Tuple,...]) -> float:
            # Unpack the grid params so that the distance_to_nearest_grid_intersection method is correct   
            self.initial_point, self.axis_1, self.axis_2 = grid_params
            total_distance = np.sum( self.distance_to_nearest_grid_intersection(point) for point in peak_locations )
            return total_distance
        
        # Find the current median separation between points to use as an initial 
        # guess at the separation of grid points
        distances = np.abs(peak_locations[1:, :] - peak_locations[:-1, :])
        median_separation = max(np.median(distances[:,0]), np.median(distances[:,1]))
        print(f"\tMedian Separation: {median_separation}")

        # Merge points that are close together as defined by the initial median separation 
        locations = []
        for peak_location in peak_locations:
            skip = False
            for location in locations:
                if np.sum(np.abs(peak_location - location)) < max(3, median_separation*0.2):
                    skip = True
            if not skip:
                locations.append(peak_location)
                
        # Recalculate the median separation after pruning out very close points 
        distances = np.abs(peak_locations[1:, :] - peak_locations[:-1, :])
        median_separation = max(np.median(distances[:,0]), np.median(distances[:,1]))
        
        
        res = minimize(
            lambda x: _summed_distance_to_grid(((x[0],x[1]), (x[2],x[3]), (x[4],x[5]))),
            [
                peak_locations[0][0],peak_locations[0][1], 
                0, median_separation, median_separation, 0.35
            ],
            method="L-BFGS-B",
        )

        return (
            (res['x'][0], res['x'][1]),
            (res['x'][2], res['x'][3]),
            (res['x'][4], res['x'][5]),
        )

    def construct_grid_points(self, shape: Tuple[int, int]) -> List[np.ndarray]:
        points = [] 
        for i in range(-shape[1],shape[1]):
            for j in range(-shape[0], shape[0]):
                point = np.array(self.initial_point) + i*np.array(self.axis_1) + j*np.array(self.axis_2) 
                if point[0] < 0 or point[0] >= shape[1]-1: 
                    continue
                if point[1] < 0 or point[1] >= shape[0]-1: 
                    continue
                points.append(point)
        return points 
        
if __name__ == "__main__":
    grid = SkewedGrid(
        (0,0),
        (0,1.0),
        (1.0, 0.0)
    )
    for point, distance in [ ((0,0), 0), ((0,1),0), ((1,0),0), ((0,1.5), 0.5), ((1.5,0),0.5), ((2.5,0.0),0.5), ((4.5, 4.5), np.sqrt(0.5))]:
        evaluated_distance =  grid.distance_to_nearest_grid_intersection(point) 
        print(f"Testing point {point}: Expected distance {distance}. Actual distance: {evaluated_distance}")
        assert evaluated_distance == distance, f"Distance to nearest grid intersection is wrong for point {point}"


    grid = SkewedGrid(
        (0,0),
        (0,4.0),
        (4.0, 0.0)
    )
    for point, distance in [ ((0,0), 0), ((0,1),1.0), ((1,0),1.0), ((0,1.5), 1.5), ((1.5,0), 1.5), ((2.5,0.0), 1.5), ((4.5, 4.5), np.sqrt(0.5))]:
        evaluated_distance =  grid.distance_to_nearest_grid_intersection(point) 
        print(f"Testing point {point}: Expected distance {distance}. Actual distance: {evaluated_distance}")
        assert evaluated_distance == distance, f"Distance to nearest grid intersection is wrong for point {point}"

