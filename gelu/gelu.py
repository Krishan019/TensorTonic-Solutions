import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    vectorized_erf = np.vectorize(math.erf)
    gel = 1/2 * x * (1 + vectorized_erf(x/np.sqrt(2)))

    return gel
