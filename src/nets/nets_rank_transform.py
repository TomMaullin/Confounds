import os
import warnings
import numpy as np
import pandas as pd
from scipy.special import ndtri

# ==========================================================================
#
# Helper function to apply the transformation to data without NaNs.
#
# --------------------------------------------------------------------------
#
# Parameters:
# - data (numpy array): Data without NaNs.
# - constant (float): Constant to be used in the transformation.
#
# --------------------------------------------------------------------------
#
# Returns:
# - Transformed data.
#
# ==========================================================================
def nets_rank_transform(data, constant):

    # Get the shape of the data and flatten data
    data_shape = data.shape
    data = data.flatten()

    # Get the unique values and argsort of the data
    unique_vals, inverse, counts, perm = unique_plus(data, return_inverse=True,
                                                     return_counts=True, 
                                                     return_perm=True)
    
    # Rank the data
    ranks = np.empty_like(perm)
    ranks[perm] = np.arange(len(data)) + 1

    # Calculate p values to transform
    n = len(data)
    p = (ranks - constant) / (n - 2 * constant + 1)

    # Use the percentile point function (inverse of CDF) from a normal distribution
    transformed_data = ndtri(p)
    
    # Create a boolean mask for values with count > 1
    mask = counts > 1
    
    # Use the mask to select indices of unique values with count > 1
    indices = np.arange(len(unique_vals))[mask]

    # Sum over the identified chunks (we need to average the transformed
    # data for data that had equal values)
    sums = np.bincount(inverse,weights=transformed_data)
    means = sums/counts
    transformed_data = means[inverse]
    
    # Reshape and return transformed data
    return(transformed_data.reshape(data_shape))



# ==========================================================================
#
# The below function is identical to the source code for numpy unique, with
# one exception; it also allows the return of the permutation from argsort.
# This is useful as it cuts computation time in half when both argsort and
# unique must be run on the same array.
#
# ==========================================================================
def unique_plus(ar, return_index=False, return_inverse=False,
                return_counts=False, return_perm=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    if return_perm:
        ret += (perm,)
    return ret