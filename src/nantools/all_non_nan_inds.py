import os 
import numpy as np
import pandas as pd
from src.memmap.MemoryMappedDF import MemoryMappedDF

# ------------------------------------------------------------------------------
#
# Helper function to get row indices where all columns have no nans
#
# ------------------------------------------------------------------------------
#
# Inputs:
#  - x (numpy array or memory mapped df): data to get nan patterns from
#  - safeMode (boolean): If true, we assume the data is too large to be read 
#                        into memory and instead read in column by column.
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

        print('hereeeeee')
        # Create an empty boolean array
        nan_array = np.zeros(x.shape[0], dtype=bool)

        # Loop through rows one by one
        for row in range(x.shape[0]):
            nan_array[row] = np.isnan(x[row, :].values).any()

        # Return result
        return(nan_array)

        
    
    