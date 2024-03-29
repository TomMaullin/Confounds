
import os
import shutil
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from lib.script_01_05 import func_01_05_gen_nonlin_conf

# from src.preproc.filter_columns_by_site import filter_columns_by_site might be useful

from src.memmap.MemoryMappedDF import MemoryMappedDF

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
# - script_01_03-4: This python script sets up a cluster instance and submits the
#                   code in script_01_05.py to the cluster as a seperate job for 
#                   each IDP. The number of nodes is not hard-coded, but instead
#                   is set by a user-defined option in cluster_cfg. This python 
#                   script absorbs the functionality of script_01_03.m,
#                   script_01_05.sh and script_01_04.sh from the matlab repo.
#
# - script_01_05: This python script is a direct translation of the matlab script
#                 func_01_05. 
#
# -------------------------------------------------------------------------------

# 03 = bash script splitting 05 
# 04 = fsl sub of scripts generated by 03
# 05 = bash script calling to func_05
# 06 = plot?


def get_p_vals_and_ve(data_dir, out_dir, nonlinear_confounds, IDPs_deconf, cluster_cfg=None):
    
    # --------------------------------------------------------------------------------
    # Handle empty configuration
    # --------------------------------------------------------------------------------
    if cluster_cfg is None:
    
        # Set new local configuration
        cluster_cfg = {'cluster_type':'local','num_nodes':1}
    
    # --------------------------------------------------------------------------------
    # Set up cluster
    # --------------------------------------------------------------------------------
    if 'cluster_type' in cluster_cfg:

        # Check if we are using a HTCondor cluster
        if cluster_cfg['cluster_type'].lower() == 'htcondor':

            # Load the HTCondor Cluster
            from dask_jobqueue import HTCondorCluster
            cluster = HTCondorCluster()

        # Check if we are using an LSF cluster
        elif cluster_cfg['cluster_type'].lower() == 'lsf':

            # Load the LSF Cluster
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()

        # Check if we are using a Moab cluster
        elif cluster_cfg['cluster_type'].lower() == 'moab':

            # Load the Moab Cluster
            from dask_jobqueue import MoabCluster
            cluster = MoabCluster()

        # Check if we are using a OAR cluster
        elif cluster_cfg['cluster_type'].lower() == 'oar':

            # Load the OAR Cluster
            from dask_jobqueue import OARCluster
            cluster = OARCluster()

        # Check if we are using a PBS cluster
        elif cluster_cfg['cluster_type'].lower() == 'pbs':

            # Load the PBS Cluster
            from dask_jobqueue import PBSCluster
            cluster = PBSCluster()

        # Check if we are using an SGE cluster
        elif cluster_cfg['cluster_type'].lower() == 'sge':

            # Load the SGE Cluster
            from dask_jobqueue import SGECluster
            cluster = SGECluster()

        # Check if we are using a SLURM cluster
        elif cluster_cfg['cluster_type'].lower() == 'slurm':

            # Load the SLURM Cluster
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster()

        # Check if we are using a local cluster
        elif cluster_cfg['cluster_type'].lower() == 'local':

            # Load the Local Cluster
            from dask.distributed import LocalCluster
            cluster = LocalCluster()

        # Raise a value error if none of the above
        else:
            raise ValueError('The cluster type, ' + cluster_cfg['cluster_type'] + ', is not recognized.')

    else:
        # Raise a value error if the cluster type was not specified
        raise ValueError('Please specify "cluster_type" in the cluster configuration.')
        
    # --------------------------------------------------------------------------------
    # Connect to client
    # --------------------------------------------------------------------------------

    # Connect to cluster
    client = Client(cluster)   
    
    # Read in number of nodes we need
    num_nodes = int(cluster_cfg['num_nodes'])
    
    # Scale the cluster
    cluster.scale(num_nodes)
    
    # --------------------------------------------------------------------------------
    # Run cluster jobs
    # --------------------------------------------------------------------------------

    # Get the number of IDPs
    num_IDPs = IDPs_deconf.shape[1]
    
    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(num_IDPs):

        # Run the i^{th} job.
        future_i = client.submit(func_01_05_gen_nonlin_conf, 
                                 data_dir, out_dir, i, nonlinear_confounds, 
                                 IDPs_deconf, pure=False)

        # Append to list 
        futures.append(future_i)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    # Delete the future objects (NOTE: This is important! If you don't delete the 
    # futures dask tries to rerun them every time you call the result function).
    del i, completed, futures, future_i
    
    
    
    