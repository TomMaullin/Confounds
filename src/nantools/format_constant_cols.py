import os 
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
#
# Find columns in the data with constant values and either fill those columns
# with a specified value or drop them, depending on the mode. Returns the 
# modified data in its original format.
#
# ------------------------------------------------------------------------------
#
# Parameters:
# - data: pandas DataFrame, numpy array, or pandas Series. The input data.
# - c: numeric, default np.NaN. The constant value to fill the columns with, if 
#      mode is 'fill'.
# - mode: str, either 'fill', 'drop', 'preserve' or 'pass'. Determines the 
#         operation on constant columns. Here is example behaviour for the 
#         each mode, for a single column, with constant=c.
#
#             fill:      [5,5,np.NaN,5,5]' --> [c,c,c,c,c]'
#             drop:      [5,5,np.NaN,5,5]' --> removed from array
#             preserve:  [5,5,np.NaN,5,5]' --> [c,c,np.NaN,c,c]'
#             pass:      [5,5,np.NaN,5,5]' --> [5,5,np.NaN,5,5]' (no change)
#             
# ------------------------------------------------------------------------------
#
# Returns:
# - The modified data, in its original format, with constant columns either 
#   filled or dropped.
#
# ------------------------------------------------------------------------------
def format_constant_cols(data, c=np.NaN, mode='fill'):
    
    # The pass mode just ignores this step of processing and returns the data
    if mode == 'pass':
        
        # Return the data, as is
        return(data)
    
    # Save the original data type
    original_type = type(data)  

    # Convert numpy array or pandas Series to DataFrame for processing
    if isinstance(data, np.ndarray):
        
        # Convert to df
        data = pd.DataFrame(data)
        
    # If series we use to_frame
    elif isinstance(data, pd.Series):
        
        # Convert to df
        data = data.to_frame()

    # Identify constant columns
    constant_columns = (data.std(ddof=0).abs() < np.finfo(np.float32).eps) | (data.isna().all())
    
    # Convert boolean Series to match DataFrame column names for alignment
    constant_column_names = constant_columns[constant_columns].index.tolist()

    # If we are in fill mode
    if mode == 'fill':
        
        # Fill constant columns with the specified value
        data[constant_column_names] = c
    
    # Else if we are preserving the patterns of nans
    elif mode == 'preserve':
        
        # Make a temporary copy of constant columns
        tmp = data[constant_column_names].values
        
        # Get the non_nan values in the constant columns
        non_nan = ~np.isnan(tmp)
        
        # Replace non_nan values with constant
        tmp[non_nan] = c
        
        # Save temporary copy back to data
        data[constant_column_names] = tmp[:,:]
        
    # Else if we are dropping
    elif mode == 'drop':
        
        # Drop constant columns
        data = data.drop(columns=constant_column_names)

    # Convert back to the original data type if necessary
    if original_type is np.ndarray:
        
        # Return np.array
        return(data.values)
    
    # If it was originally a series
    elif original_type is pd.Series:
        
        # Squeeze in case we're left with a single-column DataFrame
        return (data.squeeze())
    
    # Otherwise return
    else:
    
        # Return pandas datafame
        return(data)
