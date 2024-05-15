import os
import numpy as np
import pandas as pd
from preproc.switch_type import switch_type
from memmap.MemoryMappedDF import MemoryMappedDF
from memmap.addBlockToMmap import addBlockToMmap

# ==========================================================================
#
# Smooth IDPs. This code was adapted from the below answer on stack overflow.
#
# https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
# 
# --------------------------------------------------------------------------
#
# Parameters:
#  - time_full (MemoryMappedDF, filename or pandas df): The times at which
#                                                       data was recorded 
#                                                       (used for smoothing
#                                                       weights).
#  - IDPs (MemoryMappedDF, filename or pandas df): data to be smoothed. 
#  - sigma (scalar/float): Smoothing kernel sigma.
#  - columns (list of str): The columns of IDPs to smoothed.
#  - time_reduced_inds (np.array): The indices of the timepoints in time_full
#                                  to return smoothed values for.
#  - dtype: Output datatype (default np.float64)
#  - out_fname (str): Filename to output to (only necessary if return_df is
#                     False).
#  - return_df (boolean): If true the result is returned as a pandas df,
#                         otherwise is is saved as a memory map (default 
#                         False).
#
# --------------------------------------------------------------------------
#
# Returns:
#  - np.array: Smoothed y (Output saved to file if running parallel).
#     
# ==========================================================================
def nets_smooth_single(time_full, IDPs, sigma, columns=None, time_reduced_inds=None, 
                       null_thresh=0.6, dtype=np.float64, out_fname=None, return_df=False):

    # Switch types to save transfer costs 
    time_full = switch_type(time_full, out_type="pandas") # Only time all data is read in
    if type(IDPs) == str:
        IDPs = switch_type(IDPs,out_type="MemoryMappedDF")
       
    # Save original index and columns for outputting later
    IDPs_index_original = IDPs.index
    IDPs_columns_original = IDPs.columns

    # Get dimensions we are ouputting to
    out_dim = IDPs.shape
    
    # Get the y's we're interested in
    if type(IDPs) == MemoryMappedDF:
        IDPs_current = IDPs[:,columns]
    else:
        IDPs_current = IDPs.loc[:,columns]

    # Get the data in the form we now need
    xdata = time_full.values.flatten()
    ydata = IDPs_current.values
    xeval = xdata[time_reduced_inds]
    
    # Flatten xdata and xeval
    xdata = xdata.flatten()
    xeval = xeval.flatten()
    
    # Distance between every combination of xdata and xeval
    # each row corresponds to a value in xeval
    # each col corresponds to a value in xdata
    delta_x = xeval[:, None] - xdata
    
    # Calculate weight of every value in delta_x using Gaussian
    # Maximum weight is 1.0 where delta_x is 0
    weights = np.exp(-0.5 * ((delta_x / sigma) ** 2))
    
    # Temporarily remove zeros from ydata
    ydata_wo_nans = np.array(ydata)
    ydata_wo_nans[np.isnan(ydata)]=0
    
    # Multiply each weight by every data point, and sum over data points
    smoothed = weights @ ydata_wo_nans
    
    # Nullify the result when the total weight is below threshold
    # This happens at evaluation points far from any data
    # 1-sigma away from a data point has a weight of ~0.6
    nan_mask = weights.sum(1) < null_thresh
    smoothed[nan_mask] = np.nan
    
    # Normalize by dividing by the total weight at each evaluation point
    # Nullification above avoids divide by zero warning shere
    for k in np.arange(smoothed.shape[1]):
        
        # Get nan mask
        non_nan_mask = ~np.isnan(ydata[:,k])
        
        # Get smoothed
        smoothed[:,k] = smoothed[:,k] / weights[:,non_nan_mask].sum(1)
    
    # If we are saving the data
    if not return_df:
        
        # Indices for where to add to memmap (note: everything is transposed as these 
        # files are **much** quicker to save row by row than column by column
        indices = np.ix_([list(IDPs_columns_original).index(column) for column in columns],
                         time_reduced_inds)
    
        # Add the block to the memory map
        addBlockToMmap(out_fname,  smoothed.T, indices, (out_dim[1],out_dim[0]), dtype=dtype)

    # Otherwise return deconfounded df
    else:

        # Return result
        return(smoothed)