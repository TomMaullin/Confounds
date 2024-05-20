import os
import warnings
import numpy as np
import pandas as pd
from scipy.special import ndtri
from nets.nets_unique import nets_unique

# ==========================================================================
#
# This function applys rank-based inverse normal transformation to input 
# data.
#
# --------------------------------------------------------------------------
#
# Parameters:
#
# - data (np.array): Data (assumed to be without NaNs).
# - constant (float): Constant to be used in the transformation.
#
# --------------------------------------------------------------------------
#
# Returns:
#
# - transformed_data (np.array): Transformed data.
#
# ==========================================================================
def nets_rank_transform(data, constant):

    # Get the shape of the data and flatten data
    data_shape = data.shape
    data = data.flatten()

    # Get the unique values and argsort of the data
    unique_vals, inverse, counts, perm = nets_unique(data, return_inverse=True,
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

