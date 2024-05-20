import os 
import numpy as np
import pandas as pd
from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF

# ------------------------------------------------------------------------------
#
# Helper function to get indices where of columns in data which have no nan 
# values.
#
# ------------------------------------------------------------------------------
#
# Inputs:
#  - x (np.array or memory mapped df): data to get nan patterns from
#  - safeMode (boolean): If true, we assume the data is too large to be read 
#                        into memory and instead read in column by column.
#
# ------------------------------------------------------------------------------
#
# Returns:
#   - nan_array (np.array): Boolean array representing columns with no nan
#                           values.
#
# ------------------------------------------------------------------------------
def all_non_nan_inds(x, safeMode=False):

    # If we aren't in safe mode just read everything in.
    if not safeMode:
        
        # If the type is memory mapped
        if type(x)==MemoryMappedDF:
    
            # Get the values
            x = x[:,:].values

        return(~np.isnan(x).any(axis=1))

    # Assume we can't load all data in at once
    else:
        
        # Create an empty boolean array
        nan_array = np.zeros(x.shape[0], dtype=bool)

        # Loop through rows one by one
        for row in range(x.shape[0]):
            nan_array[row] = np.isnan(x[row, :].values).any()

        # Return result
        return(nan_array)

        
    
    