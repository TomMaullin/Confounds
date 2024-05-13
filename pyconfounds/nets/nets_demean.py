import os
import numpy as np
import pandas as pd

# ==========================================================================
#
# Demean input array x by removing the mean and scaling to unit variance.
# 
# --------------------------------------------------------------------------
#
# Parameters:
#  - x (pandas df): Input array to demean.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - x (pandas df): Demeaned array with mean removed and scaled to unit
#                   variance.
#  - 
#     
# ==========================================================================
def nets_demean(x):
    
    # Demean
    x = x.sub(x.mean(axis=0), axis=1)

    # Return x
    return(x)