import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from dask.distributed import Client, as_completed

from preproc.switch_type import switch_type
from memmap.MemoryMappedDF import MemoryMappedDF
from dasktools.connect_to_cluster import connect_to_cluster

from script_01_05 import func_01_05_gen_nonlin_conf

from logio.my_log import my_log
from logio.loading import ascii_loading_bar

# -------------------------------------------------------------------------------
# The layout of scripts 03-06 has changed substantially from the original matlab.
# -------------------------------------------------------------------------------
# 
# The original (matlab) file outline was as follows:
#
# - script_01_03: This matlab script constructed two text files each containing
#                 lists of bash commands of the form:
# 
#                      ./scripts/script_01_05_gen_nonlin_conf.sh IDP_number
#
# - script_01_05: This bash script called func_01_05 on a given IDP number.
#
# - func_01_05: This matlab script computed p values and variance explained for
#               a given IDP number.
#
# - script_01_04: This bash script submitted the scripts created by script_01_03
#                 to the cluster using fsl_sub.
#
# -------------------------------------------------------------------------------
# 
# The new outline is as follows:
#
# - script_01_03_to_4: This python script sets up a cluster instance and submits
#                      the code in script_01_05.py to the cluster as a seperate
#                      job for each IDP. The number of nodes is not hard-coded,
#                      but instead is set by a user-defined option in cluster_cfg.
#                      This python script absorbs the functionality of
#                      script_01_03.m, script_01_05.sh and script_01_04.sh from 
#                      the matlab repo.
#
# - script_01_05: This python script is a direct translation of the matlab script
#                 func_01_05. 
#
# -------------------------------------------------------------------------------


# =============================================================================
#
# This function computes and generates p-values for the variance explained by 
# a set of nonlinear confounds in a set of IDPs (which have been previously
# deconfounded using linear terms).
#
# -----------------------------------------------------------------------------
# 
# It takes the following inputs:
#
#  - data_dir (string): The directory containing the data.
#  - out_dir (string): The output directory for results to be saved to.
#  - nonlinear_confounds (string or MemoryMappedDF): The memory mapped nonlinear
#                                                    confounds.
#  - IDPs_deconf (string or MemoryMappedDF): The memory mapped IDPs, previously
#                                            deconfounded by linear terms.
#  - cluster_cfg (dict): Cluster configuration dictionary containing the type
#                        of cluster we want to run (e.g. 'local', 'slurm', 
#                        'sge',... etc) and the number of nodes we want to run 
#                        on (e.g. '12').
#  - dtype (object/string): The dtype to output data as (default np.float64).
#  - logfile (string): A html filename for the logs to be print to.
#
# -----------------------------------------------------------------------------
#
# It returns:
#
#  - p (MemoryMappedDF): Memory mapped p values.
#  - ve (MemoryMappedDF): Memory mapped variance explained.
#
# =============================================================================
def get_p_vals_and_ve(data_dir, out_dir, nonlinear_confounds, IDPs_deconf, cluster_cfg=None, 
                      dtype=np.float64, logfile=None):

    # Update log
    my_log(str(datetime.now()) +': Stage 4: Thresholding nonlinear confounds.', mode='a', filename=logfile)
    my_log(str(datetime.now()) +': Connecting to cluster...', mode='a', filename=logfile)
    
    # --------------------------------------------------------------------------------
    # Convert to memory mapped df (if not already)
    # --------------------------------------------------------------------------------
        
    # Check we have a temporary memmap directory
    if not os.path.exists(os.path.join(out_dir, 'temp_mmap')):
        os.makedirs(os.path.join(out_dir, 'temp_mmap'))
    
    # Convert input to memory mapped dataframes if it isn't already
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF', out_dir=out_dir) 
    IDPs_deconf = switch_type(IDPs_deconf, out_type='MemoryMappedDF', out_dir=out_dir) 
    
    # --------------------------------------------------------------------------------
    # Connect to the cluster
    # --------------------------------------------------------------------------------
    cluster, client = connect_to_cluster(cluster_cfg)
    
    # --------------------------------------------------------------------------------
    # Format the data appropriately
    # --------------------------------------------------------------------------------
    
    # Get the number of IDPs and non linear confounds
    num_IDPs = IDPs_deconf.shape[1]
    num_conf_nonlin = nonlinear_confounds.shape[1]
    
    # Work out columns and index for output dataframes
    indices = IDPs_deconf.columns
    columns = nonlinear_confounds.columns
    
    # Switch type to reduce transfer costs
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='filename', 
                                      fname=os.path.join(out_dir,'temp_mmap','nonlinear_confounds.dat'), 
                                      out_dir=out_dir) 
    IDPs_deconf = switch_type(IDPs_deconf, out_type='filename', 
                              fname=os.path.join(out_dir,'temp_mmap','IDPs_deconf.dat'), 
                              out_dir=out_dir) 

    # Scatter the filenames
    scattered_nonlinear_confounds = client.scatter(os.path.join(out_dir,'temp_mmap','nonlinear_confounds.dat'))
    scattered_IDPs_deconf = client.scatter(os.path.join(out_dir,'temp_mmap','IDPs_deconf.dat'))

    # Scatter the directories
    scattered_data_dir = client.scatter(data_dir)
    scattered_out_dir = client.scatter(out_dir)
    
    # --------------------------------------------------------------------------------
    # Run cluster jobs
    # --------------------------------------------------------------------------------

    # Update log
    my_log(str(datetime.now()) +': Connected to cluster.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Computing variance explained...', mode='a', filename=logfile)
    
    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(num_IDPs):

        # Run the i^{th} job.
        future_i = client.submit(func_01_05_gen_nonlin_conf, 
                                 scattered_data_dir, scattered_out_dir, i, 
                                 scattered_nonlinear_confounds, 
                                 scattered_IDPs_deconf, pure=False)

        # Append to list 
        futures.append(future_i)

    # Completed jobs
    completed = as_completed(futures)

    # Create 0% loading bar
    my_log(ascii_loading_bar(0), mode='a', filename=logfile)
    
    # Wait for results
    j = 0
    for i in completed:
        i.result()
        j = j+1
        my_log(ascii_loading_bar(100*j/num_IDPs), mode='r', filename=logfile)

    # Update log
    my_log(str(datetime.now()) +': Saving results...', mode='a', filename=logfile)
    
    # Delete the future objects (NOTE: This is important! If you don't delete the 
    # futures dask tries to rerun them every time you call the result function).
    del i, completed, futures, future_i
    
    # Create p memory mapped df
    p = np.memmap(os.path.join(out_dir,'temp_mmap', 'p.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    p = pd.DataFrame(p,index=indices,columns=columns)
    p = MemoryMappedDF(p, directory=out_dir)
    
    # Create ve memory mapped df
    ve = np.memmap(os.path.join(out_dir,'temp_mmap', 've.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    ve = pd.DataFrame(ve,index=indices,columns=columns)
    
    # Save as memory mapped dataframe
    ve = MemoryMappedDF(ve, directory=out_dir)
    
    # Remove original files
    fnames = [os.path.join(out_dir,'temp_mmap', 'p.npy'), os.path.join(out_dir,'temp_mmap', 've.npy')]
    
    # Loop through files removing each
    for fname in fnames:
        os.remove(fname)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    # Close the cluster and client
    client.close()
    client.shutdown()

    # Delete the objects for good measure
    del client, cluster
    
    # Update log
    my_log(str(datetime.now()) +': Results saved.', mode='r', filename=logfile)
    my_log(str(datetime.now()) +': Stage 4: Complete.', mode='a', filename=logfile)
    
    # Return the new memmaps
    return(p, ve)

    