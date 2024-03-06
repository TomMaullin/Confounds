import os
import numpy as np
import pandas as pd
from src.nantools.create_nan_patterns import create_nan_patterns
from src.nantools.all_non_nan_inds import all_non_nan_inds
from src.nets.nets_demean import nets_demean

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
#                   which uses a qr decomposition based approach. Note: pinv
#                   is not recommended as it is less robust to ill-conditioned
#                   matrices.
#  - demean (boolean): If true, y and conf is demeaned.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - np.array: Deconfounded y.
#     
# ==========================================================================
def nets_deconfound(y, conf, mode='svd', demean=True):
    
    # Save original index
    original_index = y.index

    # Get the indices for non-nan rows in conf
    conf_non_nan_inds = all_non_nan_inds(conf)

    # Reduce conf and y down, ignoring the nan rows for conf
    conf = conf[conf_non_nan_inds]
    y = y[conf_non_nan_inds]
    
    # Initialize empty nan dataframe
    y_deconf = pd.DataFrame(index=y.index)

    # If we are demeaning
    if demean:
        
        # Demean y and conf
        y = nets_demean(y)
        conf = nets_demean(conf)
        
    # We now need to get the nan-patterns for y
    nan_patterns_y = create_nan_patterns(y)

    # Loop through all unique nan patterns in y
    for i in nan_patterns_y:
        
        print('Deconfounding: ', i+1, '/', len(nan_patterns_y))

        # Get the pattern
        non_nan = ~np.array(nan_patterns_y[i]['pattern'],dtype=bool)

        # Check if we have at least 5 non-nan values
        if np.sum(1*non_nan) > 5:

            # Subset y to the appropriate columns
            cols = nan_patterns_y[i]['columns']

            # Get the y's we're interested in
            y_current = y[nan_patterns_y[i]['columns']]

            # Subset y and conf to the appropriate rows
            y_current = y_current[non_nan]
            conf_current = conf[non_nan]
            
            # If we are demeaning
            if demean:
                
                # Demean conf_current
                conf_current = nets_demean(conf_current)

            # Increase the precision on conf_current (just in case overflow
            # becomes a risk)
            conf_current = np.array(conf_current,dtype=np.float64)
            
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
                deconf_pred.index = y_current.index

            # Otherwise use svd
            elif mode.lower() == 'svd':

                # Multiply the left-singular values which contribute to the rank of conf
                # by the corresponding singular values to rank reduce conf
                U, S, Vt = np.linalg.svd(conf_current, full_matrices=False)

                # Get the rank of the matrix
                rank = np.sum(S > 1e-10)

                # Rank reduce U
                U = U[:, :rank] 

                # Get deconfounding variable predicted values to regress out
                deconf_pred = pd.DataFrame(U @ (U.T @ y_current))
                deconf_pred.index = y_current.index
                
            else:

                # Perform qr decomposition
                Q, R = np.linalg.qr(conf_current)
                betahat = np.linalg.pinv(R) @ (Q.T @ y_current)

                # Set computational zeros to actual zeros
                betahat[np.abs(betahat) < 1e-10] = 0

                # Get deconfounding variable predicted values to regress out
                deconf_pred = pd.DataFrame(conf_current @ betahat)
                deconf_pred.index = y_current.index

            # Get deconfounded y
            y_deconf_current = y_current - deconf_pred

            # If we are demeaning, demean y
            if demean:
                y_deconf_current = nets_demean(y_deconf_current)
            
            # Update deconfounded y (v2)
            y_deconf_current_with_nans = np.ones((len(y_deconf.index), 
                                                  len(y_deconf_current.columns)))*np.NaN
            
            # Update with current values
            y_deconf_current_with_nans[non_nan,:] = y_deconf_current.values[:,:]
            
            # Make into a dataframe with correct index and rows
            y_deconf_current_with_nans = pd.DataFrame(y_deconf_current_with_nans,
                                                     index=y_deconf.index,
                                                     columns=y_deconf_current.columns)
            
            # Horizontal concatenate
            y_deconf = pd.concat((y_deconf_current_with_nans, y_deconf), axis=1)
            
    
    # Get the list of columns in y that are also in y_deconf
    common_columns = [col for col in y.columns if col in y_deconf.columns]

    # Reorder y_deconf columns to match the order of common columns in y
    y_deconf = y_deconf[common_columns]
        
    # Remove columns where all values are NaN
    y_deconf = y_deconf.dropna(axis=1, how='all')
    
    # Restore the nan rows
    deconf_out = pd.DataFrame(index=original_index,columns=y_deconf.columns)
    deconf_out[conf_non_nan_inds] = y_deconf.values
    
    # Return result
    return(deconf_out)