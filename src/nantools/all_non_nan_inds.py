import os 
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Helper function to get row indices where all columns have no nans
# ------------------------------------------------------------------------------
def all_non_nan_inds(x):
    
    return(~np.isnan(x).any(axis=1))