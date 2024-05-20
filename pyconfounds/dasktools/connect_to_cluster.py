from dask.distributed import Client, as_completed

# ==========================================================================
#
# The below function takes in a cluster configuration dict and connects to 
# the corresponding cluster using dask. 
#
# --------------------------------------------------------------------------
#
# It takes as inputs:
#
# - cluster_cfg (dict). The dict must have the following arguments:
#                           - 'cluster_type' (str): The type of the cluster
#                                                   to load.
#                           - 'num_nodes' (int): The number of nodes to ask
#                                                for.
#                       If cluster_cfg is None, we set the type to local and
#                       the number of nodes to 1.
# 
# --------------------------------------------------------------------------
#
# It returns:
#
# - cluster: The dask cluster object.
# - client: The dask client object.
#
# ==========================================================================
def connect_to_cluster(cluster_cfg):

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

    # Return the cluster and client objects
    return(cluster, client)
    