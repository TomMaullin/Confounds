import os
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed

from src.preproc.switch_type import switch_type

from src.dask.connect_to_cluster import connect_to_cluster

from src.nets.nets_svd import nets_svd
from src.nets.nets_demean import nets_demean
from src.nets.nets_deconfound_single_iteration import nets_deconfound_single_iteration

from src.memmap.MemoryMappedDF import MemoryMappedDF
from src.nantools.all_non_nan_inds import all_non_nan_inds
from src.nantools.create_nan_patterns import create_nan_patterns

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
#  - check_nan_patterns (boolean): If true, the code will check if the
#                                  confounds can be grouped by the patterns 
#                                  of missingness they contain.
#  - dtype: Output datatype (default np.float32)
#  - cluster_cfg: dictionary containing configuration details for 
#                 parallelisation. If set to None, it is assumed no 
#                 parallelisation should be performed.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - np.array: Deconfounded y (Output saved to file if running parallel).
#     
# ==========================================================================
def nets_deconfound(y, conf, mode='nets_svd', demean=True, dtype='float64', 
                    conf_has_nans=None, check_nan_patterns=False, 
                    cluster_cfg=None):

    # Switch type to save transfer costs (we need all of conf in memory)
    if type(conf)==str:
        conf = switch_type(conf, out_type='pandas')
    if type(y)==str:
        y = switch_type(y, out_type='MemoryMappedDF')
    
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
    
        # Reduce conf down, ignoring the nan rows for conf
        conf = conf[conf_non_nan_inds]
        
        # If we have subset the data we need to demean again
        if demean:
            
            # Demean conf
            conf = nets_demean(conf)

    # Otherwise set the nan inds to none
    else:

        # Set to none
        conf_non_nan_inds = None
    
    # If we are checking unique nan patterns record the number of them
    if check_nan_patterns:
            
        # We now need to get the nan-patterns for y (we don't include
        # columns with 5 or less values).
        nan_patterns_y = create_nan_patterns(y, thresh=5)
        
        # Number of columns which meet our nan-thresholding requirements
        n_cols = len([j for i in nan_patterns_y for j in nan_patterns_y[i]['columns']])
    
    # Else, we just set n_cols to the number of columns in y for now and fix at the end
    else:
    
        # Set number of columns
        n_cols = y.shape[1]
    
    # Initialize empty nan dataframe
    y_deconf = pd.DataFrame(np.zeros((y.shape[0],n_cols),dtype=dtype),index=y.index)
    
    # Set column headers
    if check_nan_patterns:
        
        # We're only including column names for the variables that were not removed during nan pattern
        # identification.
        y_deconf.columns = [j for i in nan_patterns_y for j in nan_patterns_y[i]['columns']]
    
    else:
        
        # Copy from y
        y_deconf.columns = y.columns
    
    # If we are checking unique nan patterns record the number of them
    if check_nan_patterns:
    
        # Number of patterns
        num_patterns = len(nan_patterns_y)
    
    # Otherwise we need to loop through our variables one by one
    else:
    
        # Treating each variable as though it has its own unique pattern
        num_patterns = y.shape[1]
    
    # If we have a parallel configuration, run it.
    if cluster_cfg is not None:
        
        # Change y to memory mapped df if needed
        if type(y) != MemoryMappedDF:
            MemoryMappedDF(y).save(os.path.join(os.getcwd(),'temp_mmap','y.npz'))
        else:
            y.save(os.path.join(os.getcwd(),'temp_mmap','y.npz'))
            
        # Change conf to memory mapped df if needed
        if type(conf) != MemoryMappedDF:
            MemoryMappedDF(conf).save(os.path.join(os.getcwd(),'temp_mmap','conf.npz'))
        else:
            conf.save(os.path.join(os.getcwd(),'temp_mmap','conf.npz'))

        # Connect the cluster
        cluster, client = connect_to_cluster(cluster_cfg)

        # Print the dask dashboard address
        print(f"Dask dashboard address: {client.dashboard_link}")
        
        # Scatter the data across the workers
        scattered_y = client.scatter(os.path.join(os.getcwd(),'temp_mmap','y.npz'))
        scattered_conf = client.scatter(os.path.join(os.getcwd(),'temp_mmap','conf.npz'))
        scattered_conf_non_nan_inds = client.scatter(conf_non_nan_inds)
        mode = client.scatter(mode)
        
        # Empty futures list
        futures = []
        
        # Loop through all unique nan patterns in y
        for i in range(num_patterns):
        
            # If we have a pattern, use it
            if check_nan_patterns:
                
                # Get the pattern and columns
                non_nan = ~np.array(nan_patterns_y[i]['pattern'],dtype=bool)
                columns = nan_patterns_y[i]['columns']
        
            # Otherwise set to none
            else:
        
                # Empty pattern and current column
                non_nan = None
                columns = [y.columns[i]]
            
            # Submit a job to the local cluster
            future_i = client.submit(nets_deconfound_single_iteration, 
                                     scattered_y, scattered_conf, 
                                     scattered_conf_non_nan_inds,
                                     columns, mode, non_nan, pure=False)
            
            # Append to list 
            futures.append(future_i)
        
        # Completed jobs
        completed = as_completed(futures)
        
        # Wait for results
        j = 0
        for i in completed:
            i.result()
            j = j+1
            print('Deconfounded: ', str(j), '/', len(completed))
        
        # Delete the future objects (NOTE: see above comment in setup section).
        del i, completed, futures, future_i
        
        # Once completed, we read in the final numpy memory map
        deconf_out = np.memmap(os.path.join(os.getcwd(),'temp_mmap','y_deconf.dat'),
                               shape=(y.shape[1],y.shape[0]),dtype=np.float64) 
        deconf_out = np.asarray(deconf_out).T
    
        # Initialise output dataframe
        deconf_out = pd.DataFrame(deconf_out, index=original_index,columns=y.columns,dtype=dtype)
        
        # Drop all columns with zeros
        non_zero_cols = deconf_out.any(axis=0) 
        
        # Filter out zero columns using the mask
        deconf_out = deconf_out.loc[:, non_zero_cols]
        print('zero drop executed')
    
    # Otherwise, run in serial
    else:
        
        # Reduce conf and y down, ignoring the nan rows for conf
        if conf_has_nans:
            y = y[conf_non_nan_inds]
        
        # If we have subset the data we need to demean again
        if demean:
            
            # Demean y and conf
            y = nets_demean(y)
            
        # Loop through all unique nan patterns in y
        for i in range(num_patterns):

            # If we've already computed the nan values for this block use them
            if check_nan_patterns:
        
                # Get the y's we're interested in
                y_current = y[nan_patterns_y[i]['columns']]
                
                # Get the pattern
                non_nan = ~np.array(nan_patterns_y[i]['pattern'],dtype=bool)

            # Else we need to compute them
            else:

                # Get current y
                y_current = y[[y.columns[i]]]

                # Get non nan values
                non_nan = ~np.array(y_current.isna().astype(int).values,dtype=bool)
        
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

            # Add back in NaNs
            y_deconf[~non_nan.flatten()]=np.NaN

        
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

