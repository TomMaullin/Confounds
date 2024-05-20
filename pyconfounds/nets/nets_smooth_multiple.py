import os
import uuid
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed

from pyconfounds.preproc.switch_type import switch_type

from pyconfounds.dasktools.connect_to_cluster import connect_to_cluster

from pyconfounds.nets.nets_smooth_single import nets_smooth_single

from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF

from pyconfounds.logio.my_log import my_log
from pyconfounds.logio.loading import ascii_loading_bar

# ==========================================================================
#
# Smooths variables and timepoints in IDPs, handling missing data. This
# function takes in the data and decides how to parallelize the smoothing
# of multiple variables. Once it has decided how to chunk the computation,
# the actual smoothing itself is performed on each single chunk using
# nets_smooth_single.
# 
# --------------------------------------------------------------------------
#
# Parameters:
#  - time (MemoryMappedDF, filename or pandas df): The times at which data 
#                                                  was recorded (used for 
#                                                  smoothing weights).
#  - IDPs (MemoryMappedDF, filename or pandas df): data to be smoothed. 
#  - sigma (scalar/float): Smoothing kernel sigma.
#  - null_thresh (float): For evaluation points far from data points, the
#                         estimate will be based on very little data. If the
#                         total weight is below this threshold, return
#                         np.nan at this location. Zero means always 
#                         return an estimate. The default of 0.6 corresponds
#                         to approximately one standard deviation away from
#                         the nearest datapoint.
#  - blksize (int): The number of columns of IDPs to be smoothed. This is
#                   chosen to preserve memory.
#  - blksize_time (int): The number of timepoints to smooth (rows in IDPs). 
#                        This is also chosen to preserve memory.
#  - cluster_cfg: dictionary containing configuration details for 
#                 parallelisation. If set to None, it is assumed no 
#                 parallelisation should be performed.
#  - idx_IDPs (list/range): Indices for the columns of IDPs we are smoothing. 
#                           If set to None, all columns are smoothed.
#  - idx_time (list/range): Indices for the rows of IDPs (timepoints) we are
#                           smoothing. If set to None, all rows are smoothed.
#  - return_result (boolean): If true, results are returned as Pandas dataframe.
#                             Otherwise results are saved to a numpy memorymap
#                             named according to out_fname.
#  - out_dir (string): The output directory for results to be saved to. If set to
#                      None (default), current directory is used.
#  - out_fname (string): Filename to output results to. If set to none (default),
#                        a new output name is made using a random hash.
#  - log_file (string): Filename to output log messages to. If set to none
#                       (default), no log messages are output.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - smooth_out (pd.Dataframe): The requested smoothed data for the variables
#                               and timepoints specified.
#
# ==========================================================================
def nets_smooth_multiple(time, IDPs, sigma, null_thresh=0.6, blksize=1,
                         blksize_time=1, cluster_cfg=None, idx_IDPs=None, 
                         idx_time=None, return_result=True, out_dir=None,
                         out_fname=None, logfile=None):
    
    # ----------------------------------------------------------------------------
    # Create output filename
    # ----------------------------------------------------------------------------

    # Check if out_dir is none
    if out_dir is None:
        out_dir = os.getcwd()
        
    # Check we have a temporary memmap directory
    if not os.path.exists(os.path.join(out_dir, 'temp_mmap')):
        os.makedirs(os.path.join(out_dir, 'temp_mmap'))

    # If the output filename is none create it
    if out_fname is None:

        # IDPs smoothed filename with a hashkey added to multiple runs of the code
        # interfering with one another
        out_fname = os.path.join(out_dir, 'temp_mmap',
                                 'IDPs_smooth_' + str(uuid.uuid4()) + '.dat')
    
    # ----------------------------------------------------------------------------
    # Format data
    # ----------------------------------------------------------------------------

    # Switch type to save transfer costs
    IDPs = switch_type(IDPs, out_type='MemoryMappedDF',out_dir=out_dir)
    time = switch_type(time, out_type='pandas',out_dir=out_dir)

    # ----------------------------------------------------------------------------
    # Compute blocks
    # ----------------------------------------------------------------------------
    
    # Work out the IDPs we are looking at
    if idx_IDPs is None:
        idx_IDPs = np.arange(len(IDPs.columns))
    
    # Get the number of blocks we are breaking computation into
    num_blks = int(np.ceil(len(idx_IDPs)/blksize))
    
    # Get the indices for each block
    blocks = [idx_IDPs[i*blksize:min((i+1)*blksize,len(idx_IDPs))] for i in range(num_blks)]

    # Work out the timepoints/observations we are looking at
    if idx_time is None:
        idx_time = np.arange(time.shape[0])
    
    # Get the number of blocks we are breaking computation into
    num_blks_time = int(np.ceil(len(idx_time)/blksize_time))
    
    # Get the indices for each block
    blocks_time = [idx_time[i*blksize_time:min((i+1)*blksize_time,len(idx_time))] for i in range(num_blks_time)]

    # ----------------------------------------------------------------------------
    # Smooth
    # ----------------------------------------------------------------------------
    # If we have a parallel configuration, run it.
    if cluster_cfg is not None:
        
        # Save IDPs for distribution (note: we aren't writing over the original IDPs here)
        IDPs_fname = switch_type(IDPs, out_type='filename',out_dir=out_dir)
        time_fname = switch_type(time, out_type='filename',out_dir=out_dir)
        
        # Connect the cluster
        cluster, client = connect_to_cluster(cluster_cfg)

        # Print the dask dashboard address
        my_log(f"Dask dashboard address: {client.dashboard_link}", mode='a', filename=logfile)
        
        # Scatter the data across the workers
        scattered_time = client.scatter(time_fname)
        scattered_IDPs = client.scatter(IDPs_fname)
        scattered_sigma = client.scatter(sigma)
        scattered_null_thresh = client.scatter(null_thresh)
        scattered_blksize = client.scatter(blksize)
        scattered_return_result = client.scatter(False)
        scattered_out_dir = client.scatter(out_dir)
        scattered_out_fname = client.scatter(out_fname)

        # Empty futures list
        futures = []
        
        # Loop through each block of IDPs
        for block in blocks:

            # Loop through blocks of time/observations
            for time_block in blocks_time:
    
                # Submit a job to the local cluster
                future_i = client.submit(nets_smooth_multiple,
                                         scattered_time, scattered_IDPs,
                                         scattered_sigma, scattered_null_thresh,
                                         scattered_blksize, idx_IDPs=block, 
                                         idx_time=time_block,
                                         return_result=scattered_return_result,
                                         out_dir=scattered_out_dir,
                                         out_fname=scattered_out_fname,
                                         pure=False)
                
                # Append to list 
                futures.append(future_i)
            
        # Completed jobs
        completed = as_completed(futures)
        
        # If we are outputting to a logfile initialise it
        if logfile:
            my_log(ascii_loading_bar(0), mode='a', filename=logfile)
            
        # Wait for results
        j = 0
        for i in completed:
            i.result()
            j = j+1
            
            # Update log
            if logfile:
                my_log(ascii_loading_bar(100*j/(num_blks*num_blks_time)), mode='r', filename=logfile)
        
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
        
        # Loop through each block of IDP indexes
        for block in blocks:
            
            # Compute columns
            columns = [IDPs.columns[i] for i in block]
            
            # Perform smoothing
            nets_smooth_single(time, IDPs, sigma, columns=columns, 
                               time_reduced_inds=idx_time, 
                               null_thresh=null_thresh, dtype=np.float64, 
                               out_dir=out_dir, out_fname=out_fname)
            
    # If we are returning the result, read it back in
    if return_result:
        
        # Once completed, we read in the final numpy memory map
        smooth_out = np.memmap(out_fname, shape=(IDPs.shape[1],IDPs.shape[0]),dtype=np.float64) 
        smooth_out = np.asarray(smooth_out).T
    
        # Remove file
        os.remove(out_fname)
    
        # Initialise output dataframe
        smooth_out = pd.DataFrame(smooth_out, index=IDPs.index,columns=IDPs.columns,dtype=np.float64)
        
        # Drop all columns with zeros
        non_zero_cols = (smooth_out.abs() > 1e-8).sum(axis=0) >= 5#smooth_out.any(axis=0) 
        
        # Filter out zero columns using the mask
        smooth_out = smooth_out.loc[:, non_zero_cols]
            
        # Return result
        return(smooth_out)

