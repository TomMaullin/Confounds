import os
import numpy as np
import pandas as pd
from nantools.format_constant_cols import format_constant_cols

# ==========================================================================
#
# Normalize input array x by removing the mean and scaling to unit variance.
# 
# --------------------------------------------------------------------------
#
# Parameters:
#
#  - x (np.array/pd.Dataframe/pd.Series): Input data to normalize.
#  - constant (np.float): If x contains any constant columns, all non-NaN 
#                         values in these columns will be set to the 
#                         constant rather than NaN. e.g. below is an example
#                         of such a transform if constant=0:
#
#                            [5,5,5,5] -->  [0,0,0,0] 
#
# - mode (str): Either 'fill', 'drop', 'preserve' or 'pass'. Determines the 
#               operation on constant columns. Here is example behaviour for
#               each mode, for a single column, with constant=c.
#
#             fill:      [5,5,np.NaN,5,5]' --> [c,c,c,c,c]'
#             drop:      [5,5,np.NaN,5,5]' --> removed from array
#             preserve:  [5,5,np.NaN,5,5]' --> [c,c,np.NaN,c,c]'
#             pass:      [5,5,np.NaN,5,5]' --> [5,5,np.NaN,5,5]' (no change)
#   
# --------------------------------------------------------------------------
#
# Returns:
#
#  - x (np.array/pd.Dataframe/pd.Series): Normalized array with mean removed
#                                         and scaled to unit variance.
#     
# ==========================================================================
def nets_normalise(x, constant=np.NaN, mode='preserve'):
    
    
    # Work out if we are keeping the patterns of nans in constant columns
    if mode == 'preserve':
        keep_nan_patterns = True
    else:
        keep_nan_patterns = False
    
    # Check if x was originally a pandas object
    original_type = 'array'
    original_shape = x.shape
    
    # Check if we are looking at a series or dataframe
    if type(x)==pd.DataFrame or type(x)==pd.Series:
        
        # Record the row indices of the data
        rows = x.index
        
        # Save the original dtype
        if type(x)==pd.DataFrame:
            original_type = 'dataframe'
            
            # Record the column indices of the data
            cols = x.columns
        else:
            original_type= 'series'
        
        # Convert to numpy array
        x = x.values
        
    
    # If we haven't less than two dimensions expand
    if len(x.shape)<2:
        
        # Reshape x to have two dimensions
        x = x.reshape(x.shape+ (2-len(x.shape))*(1,))
        
    # Get the nan columns of x
    nan_cols = np.isnan(x).all(axis=0)

    # Get the constant columns of x
    constant_cols = np.abs(np.nanstd(x,axis=0)) < np.finfo(np.float32).eps
    
    # Get the remaining columns of x
    cols_to_demean = ~nan_cols & ~constant_cols

    # -------------------------------------------------------------
    # Replace the nan columns with constant
    # -------------------------------------------------------------
    
    # If there are any nan columns
    if np.any(nan_cols) & ~keep_nan_patterns:
        
        # Replace nans with constant (e.g. zero)
        x[:,nan_cols] = constant

    # -------------------------------------------------------------
    # Replace the constant columns with constant
    # -------------------------------------------------------------

    # If there are any constant columns
    if np.any(constant_cols):
    
        # Get the constant columns
        x_cols = x[:,constant_cols]

        # Replace all non-nan values with constant
        if keep_nan_patterns:

            # Get the non-nan values
            non_nans = ~np.isnan(x_cols)

            # Replace them with constant
            x_cols[non_nans] = constant

        # If we are not keeping the NaN patterns for these replace the
        # NaNs with the constant too
        else:

            # Replace all values with constant
            x_cols[:] = constant

        # Add the columns back in 
        x[:,constant_cols] = x_cols
    
    # -------------------------------------------------------------
    # Deamean the non-constant/non-nan columns
    # -------------------------------------------------------------
    
    # If there are any columns to demean demean them
    if np.any(cols_to_demean):

        # Get the columns
        x_cols = x[:,cols_to_demean] 

        # Subtract the mean along the specified dimension
        x_cols = x_cols - np.nanmean(x_cols, axis=0, keepdims=True)

        # Scale to unit variance (standard deviation = 1)
        std = np.nanstd(x_cols, axis=0, ddof=1, keepdims=True)
        x[:,cols_to_demean] = x_cols / std
        
    # Reshape back to original shape
    x = x.reshape(original_shape)

    # Convert back to correct dtype
    if original_type == 'dataframe':
        
        # Reconstruct dataframe
        x = pd.DataFrame(x)
        x.columns = cols
        x.index = rows
        
    # Convert back to correct dtype
    if original_type == 'series':
        
        # Reconstruct dataframe
        x = pd.Series(x)
        x.index = rows
        
    # Handle missing columns
    # Developer note: This step is partially already handled by the above
    # code. However, I have introduced a seperate function in order to try
    # to standardise the handling of constant columns across all functions,
    # and improve transparency of output.
    x = format_constant_cols(x, c=constant, mode=mode)
    
    # Return the result
    return(x)