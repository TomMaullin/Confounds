import os
import uuid
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
#  - blksize (int): The number of columns of y to be deconfounded at any given
#                   time. This is chosen to preserve memory.
#  - coincident (boolean): If this is set to true, we assume that all columns
#                          of y have the same patterns of NaN values. In this
#                          setting, computation can be greatly sped up as the
#                          conf matrix is only inverted once, as opposed to
#                          once for each column of y.
#  - idx_y (list/range): Indices for the columns of y we are deconfounding. 
#                        If set to None, all columns are deconfounded.
#  - return_result (boolean): If true, results are returned as MemoryMappedDFs.
#  - out_fname (string): Filename to output results to. If set to none (default),
#                        a new output name is made using a random hash.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - np.array: Deconfounded y (Output saved to file if running parallel).
#     
# ==========================================================================
def nets_deconfound_multiple(y, conf, mode='nets_svd', demean=True, dtype='float64', 
                             cluster_cfg=None, blksize=1, coincident=True, idx_y=None,
                             return_result=True, out_fname=None):
    
    # ----------------------------------------------------------------------------
    # Format data
    # ----------------------------------------------------------------------------

    # Switch type to save transfer costs (we need all of conf in memory)
    conf = switch_type(conf, out_type='pandas')
    y = switch_type(y, out_type='MemoryMappedDF')

    # ----------------------------------------------------------------------------
    # Create output filename
    # ----------------------------------------------------------------------------

    # If the output filename is none create it
    if out_fname is None:

        # Y deconfounded filename with a hashkey added to multiple runs of the code
        # interfering with one another
        out_fname = os.path.join(os.getcwd(), 'temp_mmap',
                                 'y_deconf_' + str(uuid.uuid4()) + '.dat')
    
    # ----------------------------------------------------------------------------
    # Compute blocks
    # ----------------------------------------------------------------------------
    
    # Work out the IDPs we are looking at
    if idx_y is None:
        idx_y = np.arange(len(y.columns))
    
    # Get the number of blocks we are breaking computation into
    num_blks = int(np.ceil(len(idx_y)/blksize))
    
    # Get the indices for each block
    blocks = [idx_y[i*blksize:min((i+1)*blksize,len(idx_y))] for i in range(num_blks)]
    
    # ----------------------------------------------------------------------------
    # Deconfound
    # ----------------------------------------------------------------------------
    # If we have a parallel configuration, run it.
    if cluster_cfg is not None:
        
        # Save conf and y for distribution (note: we aren't writing over the original y and conf here)
        conf_fname = switch_type(conf, out_type='filename')
        y_fname = switch_type(y, out_type='filename')
        
        # Connect the cluster
        cluster, client = connect_to_cluster(cluster_cfg)

        # Print the dask dashboard address
        print(f"Dask dashboard address: {client.dashboard_link}")
        
        # Scatter the data across the workers
        scattered_y = client.scatter(y_fname)
        scattered_conf = client.scatter(conf_fname)
        scattered_mode = client.scatter(mode)
        scattered_blksize = client.scatter(blksize)
        scattered_coincident = client.scatter(coincident)
        scattered_return_result = client.scatter(False)
        scattered_out_fname = client.scatter(out_fname)
        
        # Empty futures list
        futures = []
        
        # Loop through each block
        for block in blocks:

            # Submit a job to the local cluster
            future_i = client.submit(nets_deconfound_multiple, 
                                     scattered_y, scattered_conf, 
                                     mode=scattered_mode, 
                                     blksize=scattered_blksize, 
                                     coincident=scattered_coincident,
                                     idx_y=block, 
                                     return_result=scattered_return_result,
                                     out_fname=scattered_out_fname,
                                     pure=False)
            
            # Append to list 
            futures.append(future_i)
        
        # Completed jobs
        completed = as_completed(futures)
        
        # Wait for results
        j = 0
        for i in completed:
            i.result()
            j = j+1
            print('Deconfounded: ' + str(j) + '/' + str(num_blks))
        
        # Delete the future objects.
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

        # Check if the columns are coincident (same NaN patterns)
        if coincident:

            # Loop through each block
            for block in blocks:
                
                # Compute columns
                columns = [y.columns[i] for i in block]

                # Perform deconfounding
                nets_deconfound_single(y, conf, columns, mode=mode, demean=True, 
                                       dtype=np.float64, out_fname=out_fname)

        # If we're not running in coincident mode
        else:
            
            # Loop through columns of y
            for i in idx_y:
                
                # Perform deconfounding
                nets_deconfound_single(y, conf, [y.columns[i]], mode=mode, demean=True, 
                                       dtype=np.float64, out_fname=out_fname)
    
    # If we are returning the result, read it back in
    if return_result:
        
        # Once completed, we read in the final numpy memory map
        deconf_out = np.memmap(out_fname, shape=(y.shape[1],y.shape[0]),dtype=np.float64) 
        deconf_out = np.asarray(deconf_out).T
    
        # Remove file
        os.remove(out_fname)
    
        # Initialise output dataframe
        deconf_out = pd.DataFrame(deconf_out, index=y.index,columns=y.columns,dtype=dtype)
        
        # Drop all columns with zeros
        non_zero_cols = (deconf_out.abs() > 1e-8).sum(axis=0) >= 5#deconf_out.any(axis=0) 
        
        # Filter out zero columns using the mask
        deconf_out = deconf_out.loc[:, non_zero_cols]
            
        # Return result
        return(deconf_out)

