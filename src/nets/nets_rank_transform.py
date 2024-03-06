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
    ranks = data_series.rank(method='average')
    
    # Calculate p values to transform
    n = len(data)
    p = (ranks - constant) / (n - 2 * constant + 1)
    
    # Handle edge cases
    p = np.clip(p, 1e-5, 1 - 1e-5)  # Avoid values exactly 0 or 1 for norm.ppf
    
    # Use the percentile point function (inverse of CDF) from a normal distribution
    transformed_data = norm.ppf(p)
    
    # Return transformed data
    return(transformed_data)