from scipy import signal
from skimage.feature import blob_log
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def origin_triu(matrix: np.ndarray, origin: Tuple[int, int], k: int = 0) -> np.ndarray:
    """
    Returns the upper triangular part of a matrix considering a given origin. Similar to
    np.triu.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    origin : Tuple[int, int]
        Row and column index that defines the origin of the triangle.
    k : int, optional
        Number of diagonals to skip, defaults to 0.

    Returns
    -------
    np.ndarray
        Matrix with only the upper triangular part from the given origin.
    """
    rows, cols = matrix.shape
    r, c = origin
    mask = np.fromfunction(lambda i, j: i <= j + r - c - k, (rows, cols))
    return np.where(mask, matrix, 0)


def origin_tril(matrix: np.ndarray, origin: Tuple[int, int], k: int = 0) -> np.ndarray:
    """
    Returns the lower triangular part of a matrix considering a given origin. Similar to
    np.tril.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    origin : Tuple[int, int]
        Row and column index that defines the origin of the triangle.
    k : int, optional
        Number of diagonals to skip, defaults to 0.

    Returns
    -------
    np.ndarray
        Matrix with only the lower triangular part from the given origin.
    """
    rows, cols = matrix.shape
    r, c = origin
    mask = np.fromfunction(lambda i, j: i >= j + r - c + k, (rows, cols))
    return np.where(mask, matrix, 0)


def origin_triu_antidiagonal(
    matrix: np.ndarray, origin: Tuple[int, int], k: int = 0
) -> np.ndarray:
    """
    Returns the upper triangular part of a matrix considering a given origin and anti-diagonal. Similar to
    np.triu.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    origin : Tuple[int, int]
        Row and column index that defines the origin of the triangle.
    k : int, optional
        Number of diagonals to skip, defaults to 0.

    Returns
    -------
    np.ndarray
        Matrix with only the upper triangular part from the given origin along the anti-diagonal.
    """
    rows, cols = matrix.shape
    r, c = origin
    mask = np.fromfunction(lambda i, j: i <= -j + r + c - k, (rows, cols))
    return np.where(mask, matrix, 0)


def origin_tril_antidiagonal(
    matrix: np.ndarray, origin: Tuple[int, int], k: int = 0
) -> np.ndarray:
    """
    Returns the lower triangular part of a matrix considering a given origin and anti-diagonal. Similar to
    np.tril.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.
    origin : Tuple[int, int]
        Row and column index that defines the origin of the triangle.
    k : int, optional
        Number of diagonals to skip, defaults to 0.

    Returns
    -------
    np.ndarray
        Matrix with only the lower triangular part from the given origin along the anti-diagonal.
    """
    rows, cols = matrix.shape
    r, c = origin
    mask = np.fromfunction(lambda i, j: i >= -j + r + c + k, (rows, cols))
    return np.where(mask, matrix, 0)


def autocorrelation(image: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of an image.

    Parameters
    ----------
    image : np.ndarray
        2D input image.

    Returns
    -------
    np.ndarray
        Autocorrelation of the input image.
    """
    # Convert the image to float64 to ensure precision
    image = image.astype(np.float64)

    # Subtract the mean of the image
    image -= np.mean(image)

    # Compute the autocorrelation in the spatial domain
    autocorr = signal.correlate2d(image, image)

    return autocorr


def get_locations(
    img: np.ndarray,
    x_array: np.ndarray=None,
    y_array: np.ndarray=None,
    xlabel: str = "x (px)",
    ylabel: str = "y (px)",
    offset_px: int = 10,
    plot: bool = False,
    return_figure = False,
    max_range_locations: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze the autocorrelation of the image and return the locations of specific points 
    that should be bias triangles. 
    Use autocorrelation to find two vectors that span the location of bias triangles.
    Then use blob detection to find a single bias triangle and use that as an anchor. 
    Compute possible bias triangle locations in the image window. 

    Parameters
    ----------
    img : np.ndarray
        2D input image.
    offset_px : int, optional
        Distance to exclude in the analysis of the autocorrelation, defaults to 10.
    plot : bool, optional
        If True, plot the results, defaults to False.
    max_range_locations : int, optional
        The maximum range of locations to consider, defaults to 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple containing the anchor point, vectors spanning the structure
        of bias triangle locations, and all locations of bias triangles within image.
    """
    auto_corr_img = autocorrelation(img)

    offset_px = (
        offset_px // 2
    )  # The offset is taken in two directions so we need to half it here

    upper = origin_tril(
        origin_tril_antidiagonal(auto_corr_img, img.shape, offset_px),
        img.shape,
        offset_px,
    )
    lower = origin_triu(
        origin_tril_antidiagonal(auto_corr_img, img.shape, offset_px),
        img.shape,
        offset_px,
    )
    
    # Find the points that span the skewed rectangular pattern
    peak1 = np.argwhere(upper == upper.max())[0]
    peak2 = np.argwhere(lower == lower.max())[0]
    peaks = np.array([peak1, peak2])

    # move them to the correct point
    peaks -= np.array(auto_corr_img.shape) // 2
    peak1, peak2 = peaks
    
    
    # Do blob detection to get the absolute location and use that as an anchor
    img_norm = (img - img.min()) / (img.max() - img.min())
    blobs = blob_log(img_norm)
    if len(blobs) == 0:
        # no blobs found, use max instead
        blobs = np.argwhere(img_norm == img_norm.max())
    anchor = blobs[0]
    
    
    # Compute all locations in the image window
    all_triangles = []
    for n in range(-max_range_locations, max_range_locations):
        for m in range(-max_range_locations, max_range_locations):
            triangle = anchor[:2].copy()
            triangle += n * peak1
            triangle += m * peak2
            if (
                triangle[0] >= 0
                and triangle[0] < img.shape[0]
                and triangle[1] >= 0
                and triangle[1] < img.shape[1]
            ):
                all_triangles.append(triangle)

    all_triangles = np.array(all_triangles)

    anchor = np.array(anchor, dtype=int)
    peaks = np.array(peaks, dtype=int)
    all_triangles = np.array(all_triangles, dtype=int)

    peaks_px = peaks.copy()
    all_triangles_px = all_triangles.copy()
    if x_array is not None and y_array is not None:
        anchor = np.array([y_array[anchor[0]], x_array[anchor[1]]])
        # peaks = np.array([[y_array[peak1[0]], x_array[peak1[1]]], [y_array[peak2[0]], x_array[peak2[1]]]])
        peak_y = peaks[0] * (y_array[1] - y_array[0])
        peak_x = peaks[1] * (x_array[1] - x_array[0])
        peaks = np.array([peak_y, peak_x])
        all_triangles = np.array([[y_array[triangle[0]], x_array[triangle[1]]] for triangle in all_triangles])

    if plot or return_figure:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

        if x_array is not None and y_array is not None:
            ax1.imshow(img, origin="lower", extent=([x_array[0], x_array[-1], y_array[0], y_array[-1]]))
        else:
            ax1.imshow(img, origin="lower")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.scatter(anchor[1], anchor[0],marker='x', c="orange", s=100, label="anchor")
        ax1.scatter(
            all_triangles[:, 1],
            all_triangles[:, 0],
            c="red",
            label="suspected triangles",
        )
        ax1.set_title("original")
        ax1.legend()
        ax2.imshow(lower + upper, origin="lower")

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title("Part of autocorrelation\nused for peak detection")
        ax3.imshow(auto_corr_img, origin="lower")


        peaks_px += np.array(auto_corr_img.shape) // 2
        peak1, peak2 = peaks_px
        ax3.scatter(
            peak1[1], peak1[0], c="red", s=100, label="peaks used to infer pattern"
        )
        ax3.scatter(peak2[1], peak2[0], c="red", s=100)
        peaks_px -= np.array(auto_corr_img.shape) // 2
        peak1, peak2 = peaks_px

        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)

        ax3.set_title(f"autocorrelation")
        ax3.legend()
        if plot:
            plt.show()
        plt.close(fig)


    if return_figure:
        return anchor,peaks_px, peaks, all_triangles_px, all_triangles, fig
    else:
        return anchor,peaks_px, peaks, all_triangles_px, all_triangles


if __name__=='__main__':
    file_name = 'Ic2c4_8687.txt'
    d = np.loadtxt(
        "data_from_basel_nanowire/PSB-Examples/" + file_name,
        skiprows=3)
    values = d[:, 1:]
    lp_array = np.array(d[:, :1], dtype = float).reshape(-1)


    def read_row_from_file(filename, row_num):
        with open(filename, 'r') as f:
            lines = f.readlines()
        target_line = lines[row_num - 1]  # assuming row_num is 1-based
        values = target_line.strip().split()  # split the line into a list of values
        return np.array(values, dtype=float)  # convert the list into a numpy array


    rp_array = np.array(read_row_from_file(
        "data_from_basel_nanowire/PSB-Examples/" + file_name,
        3).reshape(-1), dtype = float)

    get_locations(values, lp_array, rp_array,'lp', 'rp', plot=True, return_figure=False)
