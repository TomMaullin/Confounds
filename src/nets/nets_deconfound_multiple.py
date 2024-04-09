import os
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed

from src.preproc.switch_type import switch_type

from src.dask.connect_to_cluster import connect_to_cluster

from src.nets.nets_svd import nets_svd
from src.nets.nets_demean import nets_demean
from src.nets.nets_deconfound_single import nets_deconfound_single

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
#  - conf (np.array): Input array to regress out from y. We assume that conf
#                     contains no nan values.
#  - mode (string): The mode of computation to use for computating betahat,
#                   current options are 'pinv' which does pinv(conf.T @ conf)
#                   @ conf.T, 'svd' which uses an svd based approach or 'qr'
#                   which uses a qr decomposition based approach, 'nets_svd'
#                   which performs an svd on conf.T @ conf. Note: pinv is not
#                   recommended as it is less robust to ill-conditioned
#                   matrices.
#  - demean (boolean): If true, y and conf is demeaned.
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
def nets_deconfound_multiple(y, conf, mode='nets_svd', demean=True, dtype='float64', 
                             cluster_cfg=None):

    # Switch type to save transfer costs (we need all of conf in memory)
    conf = switch_type(conf, out_type='pandas')
    y = switch_type(y, out_type='MemoryMappedDF')
    
    # Remove any previous versions of output (just in case)
    if os.path.exists(os.path.join(os.getcwd(),'temp_mmap','y_deconf.dat')):
        os.remove(os.path.join(os.getcwd(),'temp_mmap','y_deconf.dat'))
    
    # If we have a parallel configuration, run it.
    if cluster_cfg is not None:
        
        # Save conf and y for distribution (note: we aren't writing over the original y and conf here)
        switch_type(conf, out_type='filename', fname=os.path.join(os.getcwd(),'temp_mmap','conf.npz'))
        switch_type(y, out_type='filename', fname=os.path.join(os.getcwd(),'temp_mmap','y.npz'))
        
        # Connect the cluster
        cluster, client = connect_to_cluster(cluster_cfg)

        # Print the dask dashboard address
        print(f"Dask dashboard address: {client.dashboard_link}")
        
        # Scatter the data across the workers
        scattered_y = client.scatter(os.path.join(os.getcwd(),'temp_mmap','y.npz'))
        scattered_conf = client.scatter(os.path.join(os.getcwd(),'temp_mmap','conf.npz'))
        mode = client.scatter(mode)
        
        # Empty futures list
        futures = []
        
        # Loop through all columns of y
        for i in range(y.shape[1]):
        
            # Empty pattern and current column
            non_nan = None
            columns = [y.columns[i]]
            
            # Submit a job to the local cluster
            future_i = client.submit(nets_deconfound_single, 
                                     scattered_y, scattered_conf, 
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
            print('Deconfounded: ' + str(j) + '/' + str(y.shape[1]))
        
        # Delete the future objects (NOTE: see above comment in setup section).
        del i, completed, futures, future_i
            
        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------
    
        # Close the cluster and client
        client.close()
        client.shutdown()
    
        # Delete the objects for good measure
        del client, cluster
    
    # Otherwise, run in serial
    else:
            
        # Loop through columns of y
        for i in range(y.shape[1]):

            # Perform deconfounding
            nets_deconfound_single(y, conf, [y.columns[i]], mode='nets_svd', 
                                   non_nan=None, demean=True, dtype=np.float64)

            # Update user
            print('Deconfounded: ' + str(i) + '/' + str(y.shape[1]))
    
    # Once completed, we read in the final numpy memory map
    deconf_out = np.memmap(os.path.join(os.getcwd(),'temp_mmap','y_deconf.dat'),
                           shape=(y.shape[1],y.shape[0]),dtype=np.float64) 
    deconf_out = np.asarray(deconf_out).T

    # Initialise output dataframe
    deconf_out = pd.DataFrame(deconf_out, index=y.index,columns=y.columns,dtype=dtype)
    
    # Drop all columns with zeros
    non_zero_cols = deconf_out.any(axis=0) 
    
    # Filter out zero columns using the mask
    deconf_out = deconf_out.loc[:, non_zero_cols]
        
    # Return result
    return(deconf_out)

