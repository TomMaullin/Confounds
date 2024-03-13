
import os
import shutil
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from lib.script_01_05 import func_01_05_gen_nonlin_conf

# from src.preproc.filter_columns_by_site import filter_columns_by_site might be useful

from src.memmap.MemoryMappedDF import MemoryMappedDF

# For each IDP run script 5 basically

# Original outline was
# 03 = bash script submitting 05 
# 05 = bash script calling to func_05


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
    
    
    
    