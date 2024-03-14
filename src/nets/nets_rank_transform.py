import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

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
    
    # Convert data to a pandas Series to use the rank method
    data_series = pd.Series(data)
    
    # Rank data, treating equal elements equally by averaging their ranks
    ranks = data_series.rank(method='first').values
    
    # Calculate p values to transform
    n = len(data)
    p = (ranks - constant) / (n - 2 * constant + 1)

    # Handle edge cases
    # p = np.clip(p, 1e-5, 1 - 1e-5)  # Avoid values exactly 0 or 1 for norm.ppf
    
    # Use the percentile point function (inverse of CDF) from a normal distribution
    transformed_data = norm.ppf(p)
    
    # Get the unique values in order to handle repeats
    unique_vals, inverse, counts = np.unique(data,return_inverse=True,return_counts=True)
    
    # Create a boolean mask for values with count > 1
    mask = counts > 1
    
    # Use the mask to select indices of unique values with count > 1
    indices = np.arange(len(unique_vals))[mask]
    
    # Calculate the mean values for the selected indices
    means = np.array([np.mean(transformed_data[inverse == i]) for i in indices])
    
    # Update p with the mean values for the selected indices
    for i, idx in enumerate(indices):
        transformed_data[inverse == idx] = means[i]
    
    # Return transformed data
    return(transformed_data)