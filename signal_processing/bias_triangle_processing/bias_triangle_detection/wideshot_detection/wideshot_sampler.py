from typing import Callable, Generator, Tuple, List, Optional
import numpy as np 

from collections import defaultdict 

from bias_triangle_detection.wideshot_detection.contour_finding import evaluate_contours
from bias_triangle_detection.wideshot_detection.grid import SkewedGrid
from bias_triangle_detection.im_utils import threshold_mad 

import logging 

def get_rows(contour: List[Tuple[int, int]]) -> List[List[Tuple[int,int]]]:
    """Gets the rows of a contour. A row is defined as a set of points that have the same y coordinate. 

    Args:
        contour (List[Tuple[int, int]]): The contour to get the rows from

    Returns:
        List[List[Tuple[int,int]]]: The rows of the contour
    """    
    rows = defaultdict(list)
    for point in contour:
        rows[point[1]].append(point[0])
    
    for y_idx, row in rows.items():
        rows[y_idx] = np.linspace(min(row), max(row), max(row)-min(row)+1, dtype=int) 
    return rows

class WideshotSampler:
    def __init__(
        self, 
        scan_shape: Tuple[int, int], 
        perform_measurement: Callable[[Tuple[int,int]], Tuple[float, float] ], 
        MAX_MEASUREMENT_DISTANCE: int = 200,
        MIN_CONTOUR_LENGTH: int = 10,
        MAX_CONTOUR_LENGTH: int = 250,
        MAX_SEARCH_DISTANCE: int = 5,
        threshold: float=np.inf
    ) -> None:
        """_summary_

        Args:
            scan_shape (Tuple[int, int]): The shape of the image that is being scanned 
            perform_measurement (Callable[[Tuple[int,int]], Tuple[float, float] ]): A function that maps from pixel coordinates to a measurement value
            MAX_MEASUREMENT_DISTANCE (int, optional): The maximum allowed distance between sequential measurements. Defaults to 20.
            MIN_CONTOUR_LENGTH (int, optional): Contours that below this length are ignored. Defaults to 10.
            MAX_CONTOUR_LENGTH (int, optional): Contours above this length are ignored_. Defaults to 250.
            MAX_SEARCH_DISTANCE (int, optional): When searching for a contour with an initial point that is not a feature, a cardinal search is performed. 
                                                 This controls the arm length of the cross shape search pattern. Defaults to 5.
            threshold (float, optional): The threshold to classify a measurement as part of a triangle or not. If set to infinity a calibration scan is performed. 
        """    
        """Performs a scan across an image looking for peaks, and then performing a local scan around the peak to find a bias triangle.
        
        The algorithm works by perform a snake scan across an image. This snake scan is interrupted when a peak is found. From this peak a 
        countor is then sought. When the countour has finished being measured it is then filled in. After which the snake scan is resumed.
        
        During measurement, raw readings are converting into a binary mask. This binary mask is then used to find contours.
        The threshold is the value that is used to convert the raw readings into a binary mask. The threshold is infinity
        infinity and a calibration scan is performed. The threshold is calculated as the Median absolute deviation (MAD) 
        on the calibration scan. If a threshold is provided, the calibration scan is skipped. 
        
        Safety Guarantees:
            1. Never request a measurement outside of the scan_shape provided 
            2. Never request a measurement more than a distance of MAX_MEASUREMENT_DISTANCE pixels away from the previous measurement (counting diagonally as 1 pixel)

        Args:
           
        """        
        self.scan_shape = scan_shape
        self.perform_measurement = perform_measurement
        self.measured_values = []
        self.measurement_history = []
        self.measurement_mask = np.zeros(shape=scan_shape, dtype=bool)
        self.binary_data = np.ones(shape=scan_shape, dtype=bool) * np.inf
        
        self.calib_scan_measurement = np.zeros(shape=scan_shape)
        self.data = np.zeros(shape=scan_shape)
        self.current_position = (0,0)
        self.contours = [] 
        self.grid_width_x = max(10, scan_shape[0] // 5)
        
        self.MAX_MEASUREMENT_DISTANCE = MAX_MEASUREMENT_DISTANCE
        self.MIN_CONTOUR_LENGTH = MIN_CONTOUR_LENGTH
        self.MAX_CONTOUR_LENGTH = MAX_CONTOUR_LENGTH
        self.MAX_SEARCH_DISTANCE = MAX_SEARCH_DISTANCE
        self.threshold = threshold
        self.logger = self.configure_logger(logging.WARNING)
        self.logger.debug("Running wideshot scanning algorithm")
        self.logger.debug(f"\tScan Size: {self.scan_shape}")
        self.logger.debug(f"\tSeparation Between Line Scans: {self.grid_width_x}")


    def configure_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(log_level)
        formatter = logging.Formatter('- %(name)s - %(levelname)-8s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def sample(self, on_contour_measure: Callable[[int], None]) -> None:
        """Performs samples in the configured measurement space.

        Args:
            on_contour_measure (Callable[[int], None]): _description_

        Yields:
            Generator[Tuple[float, float], None, None]: _description_
        """        
        self.idx = 0
        if self.threshold == np.inf:
            self._calibrate_threshold(on_contour_measure)

        self.logger.info("Performing wideshot detection scan.")
        next_scan_point = (self.grid_width_x//2, 0)
        self._move_measurement_head(next_scan_point)
        self._scan(on_contour_measure, next_scan_point, evaluate_contours=True)
        self._fill_in_missing_triangles(on_contour_measure)        
        self.print_summary()    

    def _scan(self, on_contour_measure: Callable[[int], None], initial_point: Tuple[int,int], evaluate_contours: bool=False) -> None:
        """Scan across the image looking for peaks. When a peak is found, perform a contour scan around the peak.

        Args:
            on_contour_measure (Callable[[int], None]): Callback that's invoked after every contour is measured. 
                The argument is the index of the contour.
            initial_point (Tuple[int,int]): The starting point of the snake scan.
        """        
        for snake_scan_point in self.snake_scan(initial_point):
            if self.measurement_mask[snake_scan_point]:
                continue 

            if self._safe_measure(snake_scan_point) and evaluate_contours:
                self.perform_contour_scan(snake_scan_point, on_contour_measure)
                self._move_measurement_head(snake_scan_point)

    def _fill_in_missing_triangles(self, on_contour_measure) -> None:
        """The peak finding might have missed some triangles. Fit a skewed grid to different regions of the image and jump to each point on the grid. Run contour tracing around these points"""

        self.logger.info(f"Filling in missing contours.")
        window_size = (self.scan_shape[0]//2, self.scan_shape[1]//2)
        self.grid_points = []
        windows, contour_locations_by_window = self.contour_locations_in_window(window_size=window_size) 
        for window, contour_locations in zip(windows, contour_locations_by_window):
            self.logger.info(f"Fitting grid to window: {window} with {len(contour_locations)} contours.")
            if len(contour_locations) < 2:
                self.logger.debug("\tNo Contours found. Skipping window.")
                continue
            
            grid = SkewedGrid()
            try:
                grid.fit(contour_locations)
            except np.linalg.LinAlgError:
                self.logger.debug("\tFailed to fit grid! Skipping window.")
                continue

            for grid_point in grid.construct_grid_points(self.scan_shape):
                grid_point = (int(grid_point[0]), int(grid_point[1]))
                if grid_point[0] > window[0] + window_size[0] or grid_point[0] < window[0] or \
                    grid_point[1] > window[1] + window_size[1] or grid_point[1] < window[1]:
                    continue 

                self.logger.info(f"\tEvaluating point: {grid_point}")
                self.grid_points.append(grid_point)
                self.perform_contour_scan(grid_point, on_contour_measure)
        
    def print_summary(self) -> None:
        self.logger.info(f"Measurement complete.")
        self.logger.info(f"\tScan Size: {self.scan_shape}")
        self.logger.info(f"\tSeparation Between Line Scans: {self.grid_width_x}")
        self.logger.info(f"\tTotal measurements: {len(self.measurement_history)}")
        self.logger.info(f"\tGrid Size: {self.measurement_mask.size}")
        self.logger.info(f"\tPercentage of pixels measured: {len(self.measurement_history)/self.measurement_mask.size*100:.2f}%")
        self.logger.info(f"\tMax Distance Moved: {self.max_distance_moved}")

    def _calibrate_threshold(self, on_contour_measure: Callable[[int],None]) -> None:
        """Performs a snake scan across the measurement range and calculates a threshold to use for a second snake scan."""
        self.logger.info(f"Threshold not set. Performing calibration scan...")
        self._scan(on_contour_measure, (0,0), evaluate_contours=False)
        self.calib_scan_measurement = self.data[self.measurement_mask].flatten()
        self.threshold = threshold_mad(self.calib_scan_measurement)
        self.logger.info(f"Calibration Scan complete.")
        self.logger.info(f"\tThreshold: {self.threshold}")
        self.logger.info(f"\tQuantiles [0.05, 0.5, 0.95]: {np.quantile(self.calib_scan_measurement, (0.05,.5,0.95))}")
        self.logger.info(f"\tQuantile Of Score: {quantile_of_score(self.threshold, self.calib_scan_measurement)}")
            
   
    def perform_contour_scan(self, starting_point: Tuple[int,int], on_contour_measure: Callable[[int],None]) -> None:
        """Runs contour finding around a starting point and fills in the resulting contour if its within a valid size. 
        When this happens the passed callback is invoked"""
        contours = evaluate_contours(
            starting_point=starting_point,  
            measure=lambda point: self._safe_measure(point),
            bounds=self.scan_shape,
            MAX_SEARCH_DISTANCE=self.MAX_SEARCH_DISTANCE
        )
        for contour in contours:
            if len(contour) > self.MIN_CONTOUR_LENGTH and len(contour) < self.MAX_CONTOUR_LENGTH:
                self.logger.debug(f'\tFound Contour of interest, with length: {len(contour)}')
                self.idx += 1
                rows = get_rows(contour)
                triangle = []
                for y_idx in rows:
                    for x_idx in rows[y_idx]:
                        self._safe_measure((x_idx, y_idx))
                        triangle.append((x_idx, y_idx))
                self.contours.append(triangle)
                on_contour_measure(self.idx)
            else:
                self.logger.debug(f'\t\tSkipping contour of length: {len(contour)} at {starting_point}')
        
    @property 
    def contour_locations(self) -> np.ndarray: 
        if len(self.contours) == 0:
            return np.array([])
        return np.stack([np.mean(contour, axis=0) for contour in self.contours])

    @property
    def max_distance_moved(self) -> int:
        """Calculates the maximum distance moved between any two sequential measurements from the measurement history.
        """        
        max_dist = -9999
        for idx, point in enumerate(self.measurement_history[1:]):
            previous_point = self.measurement_history[idx]
            dist_to_previous_point = np.abs(point[0]-previous_point[0]) + np.abs(point[1]-previous_point[1])
            max_dist = max(max_dist, dist_to_previous_point)
        return max_dist 

    def _move_measurement_head(self, target: Tuple[int, int]) -> None:
        """Moves the measurement head to the target position.
        """        
        start_pos = self.current_position
        if np.abs(target[0]-start_pos[0]) > 0:
            for x in range(start_pos[0], target[0], np.sign(target[0]-start_pos[0])):
                self._unsafe_measure((x, start_pos[1]))
        if np.abs(target[1]-start_pos[1]) > 0:
            for y in range(start_pos[1], target[1], np.sign(target[1]-start_pos[1])):
                self._unsafe_measure((target[0], y))
    
    def _safe_measure(self, point: Tuple[int, int]) -> bool: 
        """Performs a measurement at the given point. If the target position is further from the current
        position than the MAX_MEASUREMENT_DISTANCE, then the measurement head is moved to the target position
        before performing the measurement.

        If the point has been previously measured, then the cached result is used. 

        Args:
            point (Tuple[int, int]): Point to measure

        Raises:
            ValueError: Raised if the target position is further from the current position than the MAX_MEASUREMENT_DISTANCE

        Returns:
            bool: True if the measurement is above the threshold, False otherwise
        """        
        if self.measurement_mask[point]:
            self.binary_data[point] = self.data[point] > self.threshold
            return self.data[point] > self.threshold 
        
        dist_to_previous_point = np.abs(point[0]-self.current_position[0]) + np.abs(point[1]-self.current_position[1])
        if dist_to_previous_point > self.MAX_MEASUREMENT_DISTANCE:
            self._move_measurement_head(point)
            self.binary_data[point] = self.data[point] > self.threshold
            return self.data[point] > self.threshold

        self.current_position = point
        self.measurement_history.append(self.current_position)
        self.measurement_mask[self.current_position] = True 
        
        raw_measurement = self.perform_measurement(self.current_position) 
        self.data[self.current_position] = raw_measurement
        self.measured_values.append(raw_measurement)
        self.binary_data[self.current_position] = self.data[self.current_position] > self.threshold
        return raw_measurement > self.threshold

    def _unsafe_measure(self, point: Tuple[int, int]) -> bool:
        """Performs a measurement at the given point. If the target position is further from the current 
        measurement point than the MAX_MEASUREMENT_DISTANCE, then a ValueError is raised.

        Args:
            point (Tuple[int, int]): Point to measure

        Raises:
            ValueError: Raised if the target position is further from the current position than the MAX_MEASUREMENT_DISTANCE

        Returns:
            bool: True if the measurement is above the threshold, False otherwise
        """        
        dist_to_previous_point = np.abs(point[0]-self.current_position[0]) + np.abs(point[1]-self.current_position[1])
        if dist_to_previous_point > self.MAX_MEASUREMENT_DISTANCE:
            raise ValueError(f"Measurement distance too large: {dist_to_previous_point}\nCurrent Point: {self.current_position}\nTarget Point: {point}")

        self.current_position = point
        self.measurement_history.append(self.current_position)
        self.measurement_mask[self.current_position] = True 
        
        raw_measurement = self.perform_measurement(self.current_position) 
        self.data[self.current_position] = raw_measurement
        self.measured_values.append(raw_measurement)
        self.binary_data[self.current_position] = self.data[self.current_position] > self.threshold
        return raw_measurement > self.threshold

    def snake_scan(self, initial_point: Tuple[int,int]=(0,0)) -> Generator[Tuple[int,int], None, None]:
        """Performs a snake scan across the image, starting at the initial point. 

        Args:
            initial_point (Tuple): The point to start the snake scan at 

        Yields:
            Generator[Tuple[int,int], None, None]: The next point to measure 
        """
        current_pos = initial_point 
        start_y, end_y = initial_point[1], self.scan_shape[1]
        
        y_direction = 1
        while current_pos[0] < self.scan_shape[0] - 1:
            for y in range(start_y, end_y, y_direction):
                current_pos = (current_pos[0], y)
                yield current_pos
            for x in range(current_pos[0], min(current_pos[0]+self.grid_width_x, self.scan_shape[0])):
                current_pos = (x, current_pos[1])
                yield current_pos       
            
            y_direction *= -1
            start_y = current_pos[1] 
            end_y = current_pos[1] + y_direction * self.scan_shape[1]

    def contour_locations_in_window(self, window_size: Tuple[int, int]):
        contour_locations = self.contour_locations
        contour_locations_by_window = []
        windows = []
        for x in range(0, self.scan_shape[0], window_size[0]):
            for y in range(0, self.scan_shape[1], window_size[1]):
                windows.append((x,y))
                contours_in_this_window = []
                for contour in contour_locations:
                    in_window = contour[0] >= x and contour[0] <= x + window_size[0] and \
                        contour[1] >= y and contour[1] <= y + window_size[1]
                    if not in_window:
                        continue

                    contours_in_this_window.append(contour)
                contour_locations_by_window.append(np.array(contours_in_this_window))
        return windows, contour_locations_by_window

def quantile_of_score(score: float, arr: np.ndarray) -> float:
    """Calculates what quantile the score falls in for the passed array."""
    return np.count_nonzero(arr.flatten() < score) / arr.size * 100
    

if __name__ == '__main__':
    """Example usage of the WideshotSampler.
    
    The sampler requires a perform_measurement function that takes a pixel co-ordinate and returns
    a measurement. It uses this to run the scan.
    """    
    from bias_triangle_detection.wideshot_detection.viz import plot_measurement_mask


    data = np.random.uniform(0, 255, size=(100,100))

    def perform_measurement(point: Tuple[int, int]) -> float:
            return data[point] 

    wideshot_sampler = WideshotSampler(
        scan_shape=data.shape,
        perform_measurement=perform_measurement
    )

    def plot(plot_idx: int):
        title = f"Measured Points - {np.sum(wideshot_sampler.measurement_mask)}/{wideshot_sampler.measurement_mask.size} = {np.sum(wideshot_sampler.measurement_mask)/wideshot_sampler.measurement_mask.size}"
        plot_measurement_mask(wideshot_sampler.measurement_mask, f"measurement_mask_{plot_idx}.png", title=title)

    wideshot_sampler.sample(
        on_contour_measure=plot
    )

    plot_measurement_mask(data, f"data.png")    
    plot_measurement_mask(wideshot_sampler.measurement_mask, f"measurement_mask_final.png")    