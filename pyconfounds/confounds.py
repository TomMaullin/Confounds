import os
import time 
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd

from script_01_00 import generate_initial_variables
from script_01_01 import generate_raw_confounds
from script_01_02 import generate_nonlin_confounds
from script_01_03_to_4 import get_p_vals_and_ve
from script_01_06_to_8 import threshold_ve
from script_01_09_to_12 import generate_crossed_confounds_cluster
from script_01_16 import generate_smoothed_confounds

from memmap.MemoryMappedDF import MemoryMappedDF
from memmap.read_memmap_df import read_memmap_df 

# =====================================================================================
#
# The below function is the main function for the pyconfounds library. When pyconfounds
# is run in the terminal this function is called behind the scenes. It takes in an 
# inputs yaml and creates a range of deconfounded IDPs and confound variables, stored
# as memory mapped dataframes.
#
# ------------------------------------------------------------------------------------
#
# It takes as inputs:
# - argv: Variable inputs with 0th input assumed to be string representing inputs.yml
#         file.
#
# =====================================================================================
def _main(argv=None):
    
    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Create the parser and add argument
    parser = argparse.ArgumentParser(description="PyConfounds cluster script")
    parser.add_argument('inputs_yml', type=str, nargs='?', default=os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            'pyconfounds_config.yml'), 
                        help='Path to inputs yaml file')

    # Parse the arguments
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    # If the argument is just a filename without a directory, 
    # prepend the current working directory
    if os.path.dirname(args.inputs_yml) == '':
        args.inputs_yml = os.path.join(os.getcwd(), args.inputs_yml)
    inputs_yml = args.inputs_yml

    # Change to absolute path if necessary
    inputs_yml = os.path.abspath(inputs_yml)
    
    # Load the inputs yaml file
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)

        
    # --------------------------------------------------------------------------------
    # Read directories and inputs
    # --------------------------------------------------------------------------------
    
    # Output directory (will eventually be equal to data_dir)
    out_dir = inputs['outdir']
    
    # Make directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Make memory map directory if it doesn't exist
    if not os.path.exists(os.path.join(out_dir, 'saved_memmaps')):
        os.makedirs(os.path.join(out_dir, 'saved_memmaps'))
        
    # Data directory
    data_dir = inputs['datadir']
    
    # Log file
    if 'logfile' in inputs:
        logfile = inputs['logfile']
    else:
        logfile = os.path.join(os.getcwd(), 'log.html')

    # Notify user
    print('Logs will be printed to ' + logfile)
    
    # Remove previous log file
    if os.path.exists(logfile):
        os.remove(logfile)

    # Check if MAXMEM is in the inputs
    if 'MAXMEM' in inputs:
        MAXMEM = eval(inputs['MAXMEM'])
    else:
        MAXMEM = 2**32

    # Set the cluster configuration if it wasn't provided
    if 'cluster_cfg' not in inputs:
        cluster_cfg = {'cluster_type':'local','num_nodes':12}

    # Otherwise read it in
    else:
        
        # Read in the cluster configuration
        cluster_cfg = inputs['cluster_cfg']
    
        # Make sure the number of nodes is correctly formatted
        cluster_cfg['num_nodes'] = int(cluster_cfg['num_nodes'])

    
    # --------------------------------------------------------------------------------
    # Stage 1: Generate initial variables
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        
        # Generate initial IDPs
        IDPs, nonIDPs, misc = generate_initial_variables(data_dir, tmp_dir, logfile=logfile)
        
        # Files we can reconstruct memory mapped dataframes from
        IDPs_fname = os.path.join(out_dir,'saved_memmaps','IDPs.npz')
        nonIDPs_fname = os.path.join(out_dir,'saved_memmaps','nonIDPs.npz')
        misc_fname = os.path.join(out_dir,'saved_memmaps','misc.npz')
    
        # Save the results (this is to move them outside of the temporary directory)
        IDPs.save(IDPs_fname, overwrite=True)
        nonIDPs.save(nonIDPs_fname, overwrite=True)
        misc.save(misc_fname, overwrite=True)

        # Read in saved versions
        IDPs = read_memmap_df(IDPs_fname)
        nonIDPs = read_memmap_df(nonIDPs_fname)
        misc = read_memmap_df(misc_fname)

    
    # --------------------------------------------------------------------------------
    # Stage 2: Generate raw confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
            
        # Generate raw confounds
        confounds = generate_raw_confounds(data_dir, tmp_dir, sub_ids, logfile=logfile)
        
        # Files we can reconstruct memory mapped dataframes from
        confounds_fname = os.path.join(out_dir,'saved_memmaps','confounds.npz')
    
        # Save the results (this is to move them outside of the temporary directory)
        confounds.save(confounds_fname, overwrite=True)
        
        # Read in saved confounds
        confounds = read_memmap_df(os.path.join(out_dir,'saved_memmaps','confounds.npz'))

        
    # --------------------------------------------------------------------------------
    # Stage 3: Generate nonlinear confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        
        # Set cluster configuration
        local_cluster = {'cluster_type':'local','num_nodes':12}
    
        # Generate non linear confounds and deconfound IDPs
        nonlinear_confounds, IDPs_deconf = generate_nonlin_confounds(data_dir, tmp_dir, confounds, IDPs, local_cluster, logfile=logfile)
    
        # Save the results as files we can reconstruct memory mapped dataframes from
        nonlinear_confounds_fname = os.path.join(out_dir,'saved_memmaps','nonlinear_confounds.npz')
        nonlinear_confounds.save(nonlinear_confounds_fname, overwrite=True)
        
        # Save the results as files we can reconstruct memory mapped dataframes from
        IDPs_deconf_fname = os.path.join(out_dir,'saved_memmaps','IDPs_deconf.npz')
        IDPs_deconf.save(IDPs_deconf_fname, overwrite=True)
        
        # Read in saved confounds
        nonlinear_confounds = read_memmap_df(nonlinear_confounds_fname)
    
        # Read in saved IDPs
        IDPs_deconf = read_memmap_df(IDPs_deconf_fname)

        
    # --------------------------------------------------------------------------------
    # Stage 4: Get variance explained for nonlinear confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
            
        # Set cluster configuration
        dask_cluster = {'cluster_type':'slurm','num_nodes':100}
    
        # Generate non linear confounds and deconfound IDPs
        p, ve = get_p_vals_and_ve(data_dir, tmp_dir, nonlinear_confounds, IDPs_deconf, cluster_cfg=dask_cluster, logfile=logfile)
    
        # Create filenames for memory mapped dataframes to save
        p_fname = os.path.join(out_dir,'saved_memmaps','p.npz')
        ve_fname = os.path.join(out_dir,'saved_memmaps','ve.npz')
    
        # Save the results (this is to move them outside of the temporary directory)
        p.save(p_fname)
        ve.save(ve_fname)
    
        # Read in saved p and ve
        p = read_memmap_df(p_fname)
        ve = read_memmap_df(ve_fname)


    # --------------------------------------------------------------------------------
    # Stage 5: Reduce nonlinear confounds
    # --------------------------------------------------------------------------------
    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:

        # Work out reduced nonlinear confounds
        nonlinear_confounds_reduced = threshold_ve(ve, nonlinear_confounds, tmp_dir, logfile=logfile)
        
        # Create filename for reduced nonlinear confounds 
        nonlinear_confounds_reduced_fname = os.path.join(out_dir,'saved_memmaps','nonlinear_confounds_reduced.npz')
    
        # Save memory mapped dataframe
        nonlinear_confounds_reduced.save(nonlinear_confounds_reduced_fname)

        # Read in saved
        nonlinear_confounds_reduced = read_memmap_df(nonlinear_confounds_reduced_fname)


    # --------------------------------------------------------------------------------
    # Stage 6: Generate crossed confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:

        # Set cluster configuration
        dask_cluster = {'cluster_type':'slurm','num_nodes':100}
        
        # Work out crossed confounds
        IDPs_deconf_ct, confounds_with_ct = generate_crossed_confounds_cluster(IDPs, confounds, nonlinear_confounds_reduced, 
                                                                               data_dir, tmp_dir, cluster_cfg=dask_cluster, 
                                                                               logfile=logfile)
        
        # Create filenames for memory mapped dataframes to save
        IDPs_deconf_ct_fname = os.path.join(out_dir,'saved_memmaps','IDPs_deconf_ct.npz')
        confounds_with_ct_fname = os.path.join(out_dir,'saved_memmaps','confounds_with_ct.npz')
    
        # Save memory mapped dataframes
        IDPs_deconf_ct.save(IDPs_deconf_ct_fname)
        confounds_with_ct.save(confounds_with_ct_fname)
    
        # Read in saved
        IDPs_deconf_ct = read_memmap_df(IDPs_deconf_ct_fname)
        confounds_with_ct = read_memmap_df(confounds_with_ct_fname)

    
    # --------------------------------------------------------------------------------
    # Stage 7: Generate smoothed confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:

        # Set cluster configuration
        dask_cluster = {'cluster_type':'slurm','num_nodes':100}
        
        # Get smoothed confounds
        IDPs_deconf_smooth, confounds_with_smooth = generate_smoothed_confounds(IDPs, confounds_with_ct, nonIDPs, data_dir, 
                                                                                tmp_dir, dask_cluster, logfile=logfile)
        
        # Create filenames for memory mapped dataframes to save
        IDPs_deconf_smooth_fname = os.path.join(out_dir,'saved_memmaps','IDPs_deconf_smooth.npz')
        confounds_with_smooth_fname = os.path.join(out_dir,'saved_memmaps','confounds_with_smooth.npz')
    
        # Save memory mapped dataframes
        IDPs_deconf_smooth.save(IDPs_deconf_smooth_fname)
        confounds_with_smooth.save(confounds_with_smooth_fname)
        
        # Read in saved
        IDPs_deconf_smooth = read_memmap_df(IDPs_deconf_smooth_fname)
        confounds_with_smooth = read_memmap_df(confounds_with_smooth_fname)


if __name__ == "__main__":
    _main(sys.argv[1:])