import os
import yaml 
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from pyconfounds.generate_initial_variables import generate_initial_variables
from pyconfounds.generate_raw_confounds import generate_raw_confounds
from pyconfounds.generate_nonlin_confounds import generate_nonlin_confounds
from pyconfounds.get_p_vals_and_ve_cluster import get_p_vals_and_ve_cluster
from pyconfounds.threshold_ve import threshold_ve
from pyconfounds.generate_crossed_confounds_cluster import generate_crossed_confounds_cluster
from pyconfounds.generate_smoothed_confounds import generate_smoothed_confounds

from pyconfounds.memmap.MemoryMappedDF import MemoryMappedDF
from pyconfounds.memmap.read_memmap_df import read_memmap_df 

from pyconfounds.logio.my_log import my_log

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

    # Type for output
    if 'output_dtype' in inputs:
        out_type = inputs['output_dtype']
    else:
        out_type = 'MemoryMappedDF'
        
    # Check if it is a valid type
    if out_type.lower() != 'csv':
        out_type = 'MemoryMappedDF'

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

        # Get the subject IDs
        sub_ids = IDPs.index
            
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
    
        # Generate non linear confounds and deconfound IDPs
        nonlinear_confounds, IDPs_deconf = generate_nonlin_confounds(data_dir, tmp_dir, confounds, IDPs, cluster_cfg, logfile=logfile)
    
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
    
        # Generate non linear confounds and deconfound IDPs
        p, ve = get_p_vals_and_ve_cluster(data_dir, tmp_dir, nonlinear_confounds, 
                                          IDPs_deconf, cluster_cfg=cluster_cfg, 
                                          logfile=logfile)
    
        # Create filenames for memory mapped dataframes to save
        p_fname = os.path.join(out_dir,'saved_memmaps','p.npz')
        ve_fname = os.path.join(out_dir,'saved_memmaps','ve.npz')
    
        # Save the results (this is to move them outside of the temporary directory)
        p.save(p_fname, overwrite=True)
        ve.save(ve_fname, overwrite=True)
    
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
        nonlinear_confounds_reduced.save(nonlinear_confounds_reduced_fname, overwrite=True)

        # Read in saved
        nonlinear_confounds_reduced = read_memmap_df(nonlinear_confounds_reduced_fname)


    # --------------------------------------------------------------------------------
    # Stage 6: Generate crossed confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        
        # Work out crossed confounds
        IDPs_deconf_ct, confounds_with_ct = generate_crossed_confounds_cluster(IDPs, confounds, nonlinear_confounds_reduced, 
                                                                               data_dir, tmp_dir, cluster_cfg=cluster_cfg, 
                                                                               logfile=logfile)
        
        # Create filenames for memory mapped dataframes to save
        IDPs_deconf_ct_fname = os.path.join(out_dir,'saved_memmaps','IDPs_deconf_ct.npz')
        confounds_with_ct_fname = os.path.join(out_dir,'saved_memmaps','confounds_with_ct.npz')
    
        # Save memory mapped dataframes
        IDPs_deconf_ct.save(IDPs_deconf_ct_fname, overwrite=True)
        confounds_with_ct.save(confounds_with_ct_fname, overwrite=True)
    
        # Read in saved
        IDPs_deconf_ct = read_memmap_df(IDPs_deconf_ct_fname)
        confounds_with_ct = read_memmap_df(confounds_with_ct_fname)

    
    # --------------------------------------------------------------------------------
    # Stage 7: Generate smoothed confounds
    # --------------------------------------------------------------------------------

    # Context manager temporary folder code
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        
        # Get smoothed confounds
        IDPs_deconf_smooth, confounds_with_smooth = generate_smoothed_confounds(IDPs, confounds_with_ct, nonIDPs, data_dir, 
                                                                                tmp_dir, cluster_cfg, logfile=logfile)
        
        # Create filenames for memory mapped dataframes to save
        IDPs_deconf_smooth_fname = os.path.join(out_dir,'saved_memmaps','IDPs_deconf_smooth.npz')
        confounds_with_smooth_fname = os.path.join(out_dir,'saved_memmaps','confounds_with_smooth.npz')
    
        # Save memory mapped dataframes
        IDPs_deconf_smooth.save(IDPs_deconf_smooth_fname, overwrite=True)
        confounds_with_smooth.save(confounds_with_smooth_fname, overwrite=True)
        
        # Read in saved
        IDPs_deconf_smooth = read_memmap_df(IDPs_deconf_smooth_fname)
        confounds_with_smooth = read_memmap_df(confounds_with_smooth_fname)

    
    # --------------------------------------------------------------------------------
    # Save to csv
    # --------------------------------------------------------------------------------
    if out_type.lower() == 'csv':
    
        my_log(str(datetime.now()) +': Outputting results to csv...', mode='a', filename=logfile)
    
        # Make csv directory if it doesn't exist
        if not os.path.exists(os.path.join(out_dir, 'saved_csv')):
            os.makedirs(os.path.join(out_dir, 'saved_csv'))
        
        # CSV output filenames
        IDPs_fname = os.path.join(out_dir,'saved_csv','IDPs.csv')
        nonIDPs_fname = os.path.join(out_dir,'saved_csv','nonIDPs.csv')
        misc_fname = os.path.join(out_dir,'saved_csv','misc.csv')
        confounds_fname = os.path.join(out_dir,'saved_csv','confounds.csv')
        nonlinear_confounds_fname = os.path.join(out_dir,'saved_csv','nonlinear_confounds.csv')
        IDPs_deconf_fname = os.path.join(out_dir,'saved_csv','IDPs_deconf.csv')
        p_fname = os.path.join(out_dir,'saved_csv','p.csv')
        ve_fname = os.path.join(out_dir,'saved_csv','ve.csv')
        nonlinear_confounds_reduced_fname = os.path.join(out_dir,'saved_csv','nonlinear_confounds_reduced.csv')
        IDPs_deconf_ct_fname = os.path.join(out_dir,'saved_csv','IDPs_deconf_ct.csv')
        confounds_with_ct_fname = os.path.join(out_dir,'saved_csv','confounds_with_ct.csv')
        IDPs_deconf_smooth_fname = os.path.join(out_dir,'saved_csv','IDPs_deconf_smooth.csv')
        confounds_with_smooth_fname = os.path.join(out_dir,'saved_csv','confounds_with_smooth.csv')
        
    
        # Save the IDPs
        IDPs.to_csv(IDPs_fname)
        my_log(str(datetime.now()) +': Initial IDPs saved as: ' + IDPs_fname, mode='a', filename=logfile)
        IDPs.unpersist()
        del IDPs
    
        # Save the nonIDPs
        nonIDPs.to_csv(nonIDPs_fname)
        my_log(str(datetime.now()) +': NonIDPs saved as: ' + nonIDPs_fname, mode='a', filename=logfile)
        nonIDPs.unpersist()
        del nonIDPs
        
        # Save the misc
        misc.to_csv(misc_fname)
        my_log(str(datetime.now()) +': Miscellaneous varaibles saved as: ' + misc_fname, mode='a', filename=logfile)
        misc.unpersist()
        del misc
        
        # Save the confounds
        confounds.to_csv(confounds_fname)
        my_log(str(datetime.now()) +': Initial confounds saved as: ' + confounds_fname, mode='a', filename=logfile)
        confounds.unpersist()
        del confounds
    
        # Save the nonlinear confounds
        nonlinear_confounds.to_csv(nonlinear_confounds_fname)
        my_log(str(datetime.now()) +': Nonlinear confounds saved as: ' + nonlinear_confounds_fname, mode='a', filename=logfile)
        nonlinear_confounds.unpersist()
        del nonlinear_confounds
            
        # Save the deconfounded IDPs
        IDPs_deconf.to_csv(IDPs_deconf_fname)
        my_log(str(datetime.now()) +': IDPs (deconfounded with initial confounds) saved as: ' + IDPs_deconf_fname, mode='a', filename=logfile)
        IDPs_deconf.unpersist()
        del IDPs_deconf
    
        # Save the p-values
        p.to_csv(p_fname)
        my_log(str(datetime.now()) +': P-values (for nonlinear confounds variance explained) saved as: ' + p_fname, mode='a', filename=logfile)
        p.unpersist()
        del p
    
        # Save the variance explained
        ve.to_csv(ve_fname)
        my_log(str(datetime.now()) +': Variance explained (for nonlinear confounds) saved as: ' + ve_fname, mode='a', filename=logfile)
        ve.unpersist()
        del ve
        
        # Save reduced nonlinear confounds
        nonlinear_confounds_reduced.to_csv(nonlinear_confounds_reduced_fname)
        my_log(str(datetime.now()) +': Reduced nonlinear confounds saved as: ' + nonlinear_confounds_reduced_fname, mode='a', filename=logfile)
        nonlinear_confounds_reduced.unpersist()
        del nonlinear_confounds_reduced
        
        # Save deconfounded IDPs
        IDPs_deconf_ct.to_csv(IDPs_deconf_ct_fname)
        my_log(str(datetime.now()) +': IDPs (deconfounded with nonlinear confounds) saved as: ' + IDPs_deconf_ct_fname, mode='a', filename=logfile)
        IDPs_deconf_ct.unpersist()
        del IDPs_deconf_ct
        
        # Save confounds with crossed terms
        confounds_with_ct.to_csv(confounds_with_ct_fname)
        my_log(str(datetime.now()) +': Confounds (with crossed terms) saved as: ' + confounds_with_ct_fname, mode='a', filename=logfile)
        confounds_with_ct.unpersist()
        del confounds_with_ct
    
        # Save deconfounded IDPs
        IDPs_deconf_smooth.to_csv(IDPs_deconf_smooth_fname)
        my_log(str(datetime.now()) +': IDPs (deconfounded with smoothed confounds) saved as: ' + IDPs_deconf_smooth_fname, mode='a', filename=logfile)
        IDPs_deconf_smooth.unpersist()
        del IDPs_deconf_smooth
        
        # Save confounds with smooth terms
        confounds_with_smooth.to_csv(confounds_with_smooth_fname)
        my_log(str(datetime.now()) +': Confounds (with crossed terms) saved as: ' + confounds_with_smooth_fname, mode='a', filename=logfile)
        confounds_with_smooth.unpersist()
        del confounds_with_smooth
        
        # Remove memmap folder
        if os.path.exists(os.path.join(out_dir, 'saved_memmaps')):
            shutil.rmtree(os.path.join(out_dir, 'saved_memmaps'))

    # Analysis complete message
    my_log(str(datetime.now()) +': Analysis complete!', mode='a', filename=logfile)

if __name__ == "__main__":
    _main(sys.argv[1:])
