import numpy as np
import pickle
import os


def rfc_peak_check(trace: np.ndarray) -> bool:
    """Return True (bool) if Coulomb peak(s) are present in the current trace.

    Relies on a random forext classifier to predict if Coulomb peaks are
    present. Trace is normalised before piped through the random forest
    classifier model.

    Args:
        trace (np.ndarray): Current trace of 128 pixels in length, 1D numpy
        array

    Returns:
        bool: True if Coulomb peaks are present, False if they are not
    """
    assert len(trace) == 128

    this_dir, _ = os.path.split(__file__)
    model_path = os.path.join(this_dir, "models", "rfc_peak.pkl")

    minimum_value = np.amin(trace)
    maximum_value = np.amax(trace)

    trace_norm = (trace - minimum_value) / (maximum_value - minimum_value)

    with open(model_path, "rb") as rfcpkl:
        rfc = pickle.load(rfcpkl)

    peak_found = rfc.predict(np.atleast_2d(trace_norm))

    return bool(peak_found)
