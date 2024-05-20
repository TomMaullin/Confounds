import os
import warnings
import numpy as np
import pandas as pd
from pyconfounds.nantools.format_constant_cols import format_constant_cols
from pyconfounds.nets.nets_rank_transform import nets_rank_transform

# ==========================================================================
#
# Applies a rank-based inverse normal transformation to data.
#
# --------------------------------------------------------------------------
#
# Parameters:
#
# - data (pandas dataframe): Original data, pandas dataframe.
# - constant (float, optional): Constant to be used in the transformation. 
#                               If None, method is used to set the constant.
# - method (str, optional): Method to choose the constant ('Blom', 'Tukey', 
#                          'Bliss', 'Waerden', 'SOLAR'). Ignored if constant
#                           is provided.
# - is_quantitative (bool): If True, assumes all data is quantitative and
#                           without NaN, allowing for faster execution on
#                           arrays.
# - c (np.float): If x contains any constant columns, all non-NaN values in
#                 these columns will be set to the constant rather than NaN.
#                 e.g. below is an example of such a transform if constant=0:
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
# - transformed_data (np.array): Transformed data as a numpy array.
#
# ==========================================================================
def nets_inverse_normal(data, constant=None, method=None, is_quantitative=False,
                        c=np.NaN, mode='preserve'):
    
    # Default constants for different methods
    method_constants = {
        'blom': 3/8,
        'tukey': 1/3,
        'bliss': 1/2,
        'waerden': 0,
        'solar': 0,
    }

    # Check if constant is none
    if constant is None:
        
        # Check if method is available
        if method is not None:
            
            # Convert string to lower case
            method = method.lower()
            
            # Check if method in 'method_constants'
            if method in method_constants:
                
                # Get constant for method
                constant = method_constants[method]
                
            # If unknown method specified.
            else:
                
                # Raise error
                raise ValueError("Unknown method. Use 'Blom', 'Tukey', 'Bliss', 'Waerden', or 'SOLAR'.")
        else:
            
            # Default to Blom if nothing is specified
            constant = 3/8  

    # Record the row and column indices of the data
    cols = data.columns
    rows = data.index
            
    # Transform data to a numpy array for manipulation
    data = np.array(data, dtype=np.float64)

    # Check if the data is quantitative
    if is_quantitative:
        
        # If data is quantitative and without NaN, process it directly
        transformed_data = nets_rank_transform(data, constant)
        
    else:
        
        # If data may contain NaN, process each column separately, handling NaNs
        if data.ndim == 1:
            
            # Make sure data is not 1D
            data = data.reshape(-1, 1)  
            
        # Initialise empty array with nans
        transformed_data = np.nan * np.ones(data.shape)  
        
        # Loop through columns transforming
        for col in range(data.shape[1]):
            
            # Get column data
            column_data = data[:, col]
            
            # Find non-nan locations
            valid_data_mask = ~np.isnan(column_data)
            
            # Get the non-nan data
            valid_data = column_data[valid_data_mask]
            
            # Tranform the data
            transformed_column_data = nets_rank_transform(valid_data, constant)
            
            # Output transformed data
            transformed_data[valid_data_mask, col] = transformed_column_data

    # Transform back to dataframe
    transformed_data = pd.DataFrame(transformed_data, columns = cols, index = rows)
    
    # Handle missing columns
    transformed_data = format_constant_cols(transformed_data, c=c, mode=mode)
    
    # Return result
    return(transformed_data) 
