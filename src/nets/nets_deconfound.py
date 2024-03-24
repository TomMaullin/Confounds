import os
import numpy as np
import pandas as pd
from src.nets.nets_svd import nets_svd
from src.nets.nets_demean import nets_demean
from src.memmap.MemoryMappedDF import MemoryMappedDF
from src.nantools.all_non_nan_inds import all_non_nan_inds
from src.nantools.create_nan_patterns import create_nan_patterns

import time
# ==========================================================================
#
# Regresses conf out of y, handling missing data. Demeans data unless
# specified.
# 
# --------------------------------------------------------------------------
#
# Parameters:
#  - y (np.array): Input array to regress confounds out from.
#  - conf (np.array): Input array to regress out from y.
#  - mode (string): The mode of computation to use for computating betahat,
#                   current options are 'pinv' which does pinv(conf.T @ conf)
#                   @ conf.T, 'svd' which uses an svd based approach or 'qr'
#                   which uses a qr decomposition based approach, 'nets_svd'
#                   which performs an svd on conf.T @ conf. Note: pinv is not
#                   recommended as it is less robust to ill-conditioned
#                   matrices.
#  - demean (boolean): If true, y and conf is demeaned.
#  - conf_has_nans (boolean): If true, the code will check if the confounds
#                             contain nan values
#  - dtype: Output datatype (default np.float32)
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - np.array: Deconfounded y.
#     
# ==========================================================================
def nets_deconfound(y, conf, mode='nets_svd', demean=True, dtype='float64', conf_has_nans=None):
    
    # Save original index
    original_index = y.index
    
    # Check if confounds have NaNs
    if conf_has_nans is None:
        
        # If the type is memory mapped
        if type(conf)==MemoryMappedDF:
    
            # Work out if the confounds have nans
            conf_has_nans = conf[:,:].isna().sum().sum()
    
        else:
            # Work out if the confounds have nans
            conf_has_nans = conf.isna().sum().sum()
    
    # If the confounds have nans
    if conf_has_nans:
        
        # If the type is memory mapped
        if type(conf)==MemoryMappedDF:
            
            # We are trying to avoid reading everything in at once
            conf_non_nan_inds = all_non_nan_inds(conf, safeMode=True)
            
        else:
            
            # Otherwise, we can get the indices for non-nan rows in conf directly
            conf_non_nan_inds = all_non_nan_inds(conf)
    
        # Reduce conf and y down, ignoring the nan rows for conf
        conf = conf[conf_non_nan_inds]
        y = y[conf_non_nan_inds]
        
        # If we have subset the data we need to demean again
        if demean:
            
            # Demean y and conf
            y = nets_demean(y)
            conf = nets_demean(conf)
    
        
    # We now need to get the nan-patterns for y (we don't include
    # columns with 5 or less values).
    nan_patterns_y = create_nan_patterns(y, thresh=5)

    # Number of columns which meet our nan-thresholding requirements
    n_cols = len([j for i in nan_patterns_y for j in nan_patterns_y[i]['columns']])
    
    # Initialize empty nan dataframe
    y_deconf = pd.DataFrame(np.zeros((y.shape[0],n_cols),dtype=dtype),index=y.index)
    
    # We're only including column names for the variables that were not removed during nan pattern
    # identification.
    y_deconf.columns = [j for i in nan_patterns_y for j in nan_patterns_y[i]['columns']]
    
    # Loop through all unique nan patterns in y
    for i in nan_patterns_y:
        
        t1 = time.time()
        print('Deconfounding: ', i+1, '/', len(nan_patterns_y))
    
        # Get the pattern
        non_nan = ~np.array(nan_patterns_y[i]['pattern'],dtype=bool)
    
        # Get the y's we're interested in
        y_current = y[nan_patterns_y[i]['columns']]
    
        # Subset y and conf to the appropriate rows
        y_current = y_current[non_nan]
        conf_current = conf[non_nan]
        
        # Save y index and columns
        y_index = y_current.index
        y_columns = y_current.columns
        
        # If we are demeaning
        if demean:
            
            # Demean conf_current
            conf_current = nets_demean(conf_current)
            
        # We don't want to work on views of the data as it will slow the computation
        conf_current = np.array(conf_current.values)
        y_current = np.array(y_current.values)
        
        # Check if we are using psuedo inverse
        if mode.lower() == 'pinv':
    
            # Regress conf out of y_current - we perform the pseudo inverse on
            # conf^T @ conf as we expect the number of columns to be much(!) less
            # than the number of rows and thus this ends up being more numerically
            # stable than trying to invert, or approximately invert, conf itself.
            betahat = np.linalg.pinv(conf_current.T @ conf_current) @ conf_current.T @ y_current
    
            # Set computational zeros to actual zeros
            betahat[np.abs(betahat) < 1e-10] = 0
    
            # Get deconfounding variable predicted values to regress out
            deconf_pred = pd.DataFrame(conf_current @ betahat)
            deconf_pred.index = y_index
            deconf_pred.columns = y_columns
    
        # Otherwise use svd
        elif mode.lower() == 'nets_svd':
            
            # Multiply the left-singular values which contribute to the rank of conf
            # by the corresponding singular values to rank reduce conf
            U, S, _ = nets_svd(conf_current, reorder=False)
            
            # Rank reduce U and reduce datatype as only need to multiply
            # U = U[:, S < 1e-10]
            
            # Get deconfounding variable predicted values to regress out
            deconf_pred = pd.DataFrame(U @ (U.T @ y_current))
            deconf_pred.index = y_index
            deconf_pred.columns = y_columns
    
        # Otherwise use svd
        elif mode.lower() == 'svd':
            
            # Multiply the left-singular values which contribute to the rank of conf
            # by the corresponding singular values to rank reduce conf
            U, S, _ = np.linalg.svd(conf_current, full_matrices=False)
            
            # Get the rank of the matrix
            rank = np.sum(S > 1e-10)
            
            # Rank reduce U and reduce datatype as only need to multiply
            U = U[:, :rank]
            
            # Get deconfounding variable predicted values to regress out
            deconf_pred = pd.DataFrame(U @ (U.T @ y_current))
            deconf_pred.index = y_index
            deconf_pred.columns = y_columns
            
        else:
    
            # Perform qr decomposition
            Q, R = np.linalg.qr(conf_current)
            betahat = np.linalg.pinv(R) @ (Q.T @ y_current)
    
            # Set computational zeros to actual zeros
            betahat[np.abs(betahat) < 1e-10] = 0
    
            # Get deconfounding variable predicted values to regress out
            deconf_pred = pd.DataFrame(conf_current @ betahat)
            deconf_pred.index = y_index
            deconf_pred.columns = y_columns
            
        # Get deconfounded y
        y_deconf_current = pd.DataFrame(y_current, index=y_index, columns=y_columns) - deconf_pred
    
        # If we are demeaning, demean y
        if demean:
            y_deconf_current = nets_demean(y_deconf_current)
        
        # Update deconfounded y 
        y_deconf_current_with_nans = np.ones((len(y_deconf.index), 
                                              len(y_deconf_current.columns)))*np.NaN
        
        # Update with current values
        y_deconf_current_with_nans[non_nan,:] = y_deconf_current.values[:,:]
        
        # Make into a dataframe with correct index and rows
        y_deconf_current_with_nans = pd.DataFrame(y_deconf_current_with_nans,
                                                 index=y_deconf.index,
                                                 columns=y_deconf_current.columns)
        
        # Horizontal concatenate
        y_deconf.update(y_deconf_current_with_nans)
    
        t2 = time.time()

        print('iteration time: ', t2-t1)
    
    # Get the list of columns in y that are also in y_deconf
    common_columns = [col for col in y.columns if col in y_deconf.columns]
    
    # Reorder y_deconf columns to match the order of common columns in y
    y_deconf = y_deconf[common_columns]
        
    # Initialise output dataframe
    deconf_out = pd.DataFrame(index=original_index,columns=y_deconf.columns,dtype=dtype)
    
    # Restore the nan rows
    if conf_has_nans:
        deconf_out[conf_non_nan_inds] = np.array(y_deconf.values,dtype=dtype)
    else:
        deconf_out[:] = np.array(y_deconf.values,dtype=dtype)

    # Return result
    return(deconf_out)

