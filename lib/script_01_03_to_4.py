import os
import shutil
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed

from src.preproc.switch_type import switch_type
from src.memmap.MemoryMappedDF import MemoryMappedDF
from src.dask.connect_to_cluster import connect_to_cluster

from lib.script_01_05 import func_01_05_gen_nonlin_conf

# from src.preproc.filter_columns_by_site import filter_columns_by_site might be useful


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



def get_p_vals_and_ve(data_dir, nonlinear_confounds, IDPs_deconf, cluster_cfg=None, dtype=np.float64):
    
    # --------------------------------------------------------------------------------
    # Convert to memory mapped df (if not already)
    # --------------------------------------------------------------------------------
        
    # Convert input to memory mapped dataframes if it isn't already
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF')
    IDPs_deconf = switch_type(IDPs_deconf, out_type='MemoryMappedDF')
    
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
                                      fname=os.path.join(os.getcwd(),'temp_mmap','nonlinear_confounds.dat'))
    IDPs_deconf = switch_type(IDPs_deconf, out_type='filename', 
                              fname=os.path.join(os.getcwd(),'temp_mmap','IDPs_deconf.dat'))

    # Scatter the filenames
    scattered_nonlinear_confounds = client.scatter(os.path.join(os.getcwd(),'temp_mmap','nonlinear_confounds.dat'))
    scattered_IDPs_deconf = client.scatter(os.path.join(os.getcwd(),'temp_mmap','IDPs_deconf.dat'))

    # Scatter the directories
    scattered_data_dir = client.scatter(data_dir)
    
    # --------------------------------------------------------------------------------
    # Run cluster jobs
    # --------------------------------------------------------------------------------
    
    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(num_IDPs):

        # Run the i^{th} job.
        future_i = client.submit(func_01_05_gen_nonlin_conf, 
                                 scattered_data_dir, i, 
                                 scattered_nonlinear_confounds, 
                                 scattered_IDPs_deconf, pure=False)

        # Append to list 
        futures.append(future_i)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    j = 0
    for i in completed:
        i.result()
        j = j+1
        print('Computed p-values and variance explained for: ' + str(j) + '/' + str(num_IDPs))

    # Delete the future objects (NOTE: This is important! If you don't delete the 
    # futures dask tries to rerun them every time you call the result function).
    del i, completed, futures, future_i
    
    # Create p1 memory mapped df
    p1 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 'p1.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    p1 = pd.DataFrame(p1,index=indices,columns=columns)
    p1 = MemoryMappedDF(p1)
    
    # Create p2 memory mapped df
    p2 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 'p2.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    p2 = pd.DataFrame(p2,index=indices,columns=columns)
    p2 = MemoryMappedDF(p2)
    
    # Create p3 memory mapped df
    p3 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 'p3.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    p3 = pd.DataFrame(p3,index=indices,columns=columns)
    p3 = MemoryMappedDF(p3)
    
    # Create ve1 memory mapped df
    ve1 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 've1.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    ve1 = pd.DataFrame(ve1,index=indices,columns=columns)
    ve1 = MemoryMappedDF(ve1)
    
    # Create ve2 memory mapped df
    ve2 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 've2.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    ve2 = pd.DataFrame(ve2,index=indices,columns=columns)
    ve2 = MemoryMappedDF(ve2)
    
    # Create ve3 memory mapped df
    ve3 = np.memmap(os.path.join(os.getcwd(),'temp_mmap', 've3.npy'),dtype=dtype,
                   shape=(num_IDPs, num_conf_nonlin),mode='r')[:,:]
    ve3 = pd.DataFrame(ve3,index=indices,columns=columns)
    ve3 = MemoryMappedDF(ve3)
    
    # Remove original files
    fnames = [os.path.join(os.getcwd(),'temp_mmap', 'p1.npy'), os.path.join(os.getcwd(),'temp_mmap', 'p2.npy'),
              os.path.join(os.getcwd(),'temp_mmap', 'p3.npy'), os.path.join(os.getcwd(),'temp_mmap', 've1.npy'),
              os.path.join(os.getcwd(),'temp_mmap', 've2.npy'), os.path.join(os.getcwd(),'temp_mmap', 've3.npy')]
    
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
    
    # Return the new memmaps
    return(p1, p2, p3, ve1, ve2, ve3)

    