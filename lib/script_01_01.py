import os
import shutil
import numpy as np
import pandas as pd

from src.nets.nets_load_match import nets_load_match 
from src.nets.nets_normalise import nets_normalise

from src.duplicate import duplicate_categorical, duplicate_demedian_norm_by_site
from src.preproc import datenum, days_in_year

from src.memmap import MemoryMappedDF


def generate_raw_confounds(data_dir, sub_ids):

    # Developer Note: The sorting that used to be at the start of script_01_01 has been
    # relocated to the end of script 01_00 as it needed variables from that script that
    # aren't used here
    
    # ----------------------------------------------------------------------------------
    # Load site ID and construct factors
    # ----------------------------------------------------------------------------------

    # Data types for IDs
    dtypes = {0: 'int32', 1: 'Int16'}

    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)

    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)

    # Initialize indSite as a list to hold the indices
    inds_per_site = []

    # Loop over each value in site ids
    for site_id in unique_site_ids:

        # Find the indices where all elements in a row of siteDATA match the current valueSite
        # Note: This assumes siteDATA and siteValues have compatible shapes or values for comparison
        indices = np.where((site_ids == site_id).all(axis=1))[0]

        # Append the found indices to the indSite list
        inds_per_site.append(indices)

    # Delete the indices
    del indices


    # ----------------------------------------------------------------------------------
    # Construct confounds for between sites
    # ----------------------------------------------------------------------------------

    # Get the number of rows in ALL_IDs
    n = len(sub_ids)
    
    # Initialize names_site list and conf_site matrix
    names_site = []
    conf_site = np.zeros((n, len(inds_per_site)-1))

    # Subjects from Site 1 will have -1 in all site confounds
    conf_site[inds_per_site[0], :] = -1

    # Subjects for the other sites will have -1 in their corresponding column
    # Value by default is 0
    for i in range(1, len(inds_per_site)):
        conf_site[inds_per_site[i], i-1] = 1
        names_site.append(f'Site_1_vs_{i+1}')

    # Normalize conf_site using the nets_normalise function
    conf_site = nets_normalise(conf_site)
    conf_site[np.isnan(conf_site)] = 0

    # Make into dataframe
    conf_site = pd.DataFrame(conf_site)
    conf_site.columns = names_site
    
    
    # ----------------------------------------------------------------------------------
    # Construct dummy confounds for categorical variables
    # ----------------------------------------------------------------------------------

    # Construct dummy variables for the following
    conf_sex          = duplicate_categorical('SEX',         sub_ids, inds_per_site, data_dir)
    conf_batch        = duplicate_categorical('BATCH',       sub_ids, inds_per_site, data_dir)
    conf_cmrr         = duplicate_categorical('CMRR',        sub_ids, inds_per_site, data_dir)
    conf_protocol     = duplicate_categorical('PROTOCOL',    sub_ids, inds_per_site, data_dir)
    conf_service_pack = duplicate_categorical('SERVICEPACK', sub_ids, inds_per_site, data_dir)
    conf_scan_events  = duplicate_categorical('SCANEVENTS',  sub_ids, inds_per_site, data_dir)
    conf_flipped_swi  = duplicate_categorical('FLIPPEDSWI',  sub_ids, inds_per_site, data_dir)
    conf_fst2         = duplicate_categorical('FST2',        sub_ids, inds_per_site, data_dir)
    conf_new_eddy     = duplicate_categorical('NEWEDDY',     sub_ids, inds_per_site, data_dir)
    conf_scaling      = duplicate_categorical('SCALING',     sub_ids, inds_per_site, data_dir)
    conf_time_points  = duplicate_categorical('TIMEPOINTS',  sub_ids, inds_per_site, data_dir)
    
    # Concatenate all the DataFrames/Series horizontally
    categorical_IDPs = pd.concat([
        conf_sex.reset_index(drop=True), 
        conf_batch.reset_index(drop=True),
        conf_cmrr.reset_index(drop=True),
        conf_protocol.reset_index(drop=True),
        conf_service_pack.reset_index(drop=True),
        conf_scan_events.reset_index(drop=True),
        conf_flipped_swi.reset_index(drop=True),
        conf_fst2.reset_index(drop=True),
        conf_new_eddy.reset_index(drop=True),
        conf_scaling.reset_index(drop=True),
        conf_time_points.reset_index(drop=True)
    ], axis=1)

    # Set row indices on dataframe
    categorical_IDPs.index = sub_ids
    
    
    # ----------------------------------------------------------------------------------
    # Construct dummy confounds for continuous variables
    # ----------------------------------------------------------------------------------

    # Construct dummy variables for the following
    conf_head_motion         = duplicate_demedian_norm_by_site('HEADMOTION',       sub_ids, inds_per_site, data_dir)
    conf_head_motion_st      = duplicate_demedian_norm_by_site('HEADMOTIONST',     sub_ids, inds_per_site, data_dir)
    conf_head_size           = duplicate_demedian_norm_by_site('HEADSIZE',         sub_ids, inds_per_site, data_dir)
    conf_table_pos           = duplicate_demedian_norm_by_site('TABLEPOS',         sub_ids, inds_per_site, data_dir)
    conf_dvars               = duplicate_demedian_norm_by_site('DVARS',            sub_ids, inds_per_site, data_dir)
    conf_eddy_qc             = duplicate_demedian_norm_by_site('EDDYQC',           sub_ids, inds_per_site, data_dir)
    conf_struct_head_motion  = duplicate_demedian_norm_by_site('STRUCTHEADMOTION', sub_ids, inds_per_site, data_dir)
    conf_age                 = duplicate_demedian_norm_by_site('AGE',              sub_ids, inds_per_site, data_dir)
    conf_te                  = duplicate_demedian_norm_by_site('TE',               sub_ids, inds_per_site, data_dir) 

    # Concatenate all the DataFrames/Series horizontally
    continuous_IDPs = pd.concat([
        conf_age.reset_index(drop=True),
        conf_head_size.reset_index(drop=True),
        conf_te.reset_index(drop=True),
        conf_struct_head_motion.reset_index(drop=True),
        conf_dvars.reset_index(drop=True),
        conf_head_motion.reset_index(drop=True),
        conf_head_motion_st.reset_index(drop=True), 
        conf_table_pos.reset_index(drop=True),
        conf_eddy_qc.reset_index(drop=True),
    ], axis=1)
    

    # ----------------------------------------------------------------------------------
    # Construct dummy confounds for age-sex interaction
    # ----------------------------------------------------------------------------------

    # Initialize conf_age_sex with zeros of the same shape as conf_age
    conf_age_sex = pd.DataFrame(np.zeros_like(conf_age))

    # Loop over columns in conf_age
    for i in range(conf_age.shape[1]):

        # Find indices where confAge is not zero
        ind_non_zero = np.where(conf_age.iloc[:, i] != 0)[0]

        # Apply nets_normalise to the product of confAge and confSex for non-zero age indices
        conf_age_sex.iloc[ind_non_zero, i] = nets_normalise(conf_age.iloc[ind_non_zero, i] * conf_sex.iloc[ind_non_zero, i])

        # Replace NaN values with 0, if any
        conf_age_sex[np.isnan(conf_age_sex)] = 0

    # Generate names for AgeSex per site
    names_age_sex = [f'AgeSex_Site_{j}' for j in range(1, len(inds_per_site) + 1)]

    # Set column names for conf_age_sex
    conf_age_sex.columns = names_age_sex
    
    # Create confounds dataframe
    confounds = pd.concat([
        conf_site.reset_index(drop=True),
        categorical_IDPs.reset_index(drop=True),
        continuous_IDPs.reset_index(drop=True),
        conf_age_sex.reset_index(drop=True)
    ], axis=1)
    
    # Save the index
    confounds.index = sub_ids
    
    # Get confounds memory mapped dataframe
    confounds = MemoryMappedDF(confounds)
    
    # Quick access of site variables
    confounds.set_group('SITE', conf_site.columns.tolist())
    
    # Add groups of categorical variable names for easy access
    confounds.set_group('SEX', conf_sex.columns.tolist())
    confounds.set_group('BATCH', conf_batch.columns.tolist())
    confounds.set_group('CMRR', conf_cmrr.columns.tolist())
    confounds.set_group('PROTOCOL', conf_protocol.columns.tolist())
    confounds.set_group('SERVICE_PACK', conf_service_pack.columns.tolist())
    confounds.set_group('SCAN_EVENTS', conf_scan_events.columns.tolist())
    confounds.set_group('FLIPPED_SWI', conf_flipped_swi.columns.tolist())
    confounds.set_group('FS_T2', conf_fst2.columns.tolist())
    confounds.set_group('NEW_EDDY', conf_new_eddy.columns.tolist())
    confounds.set_group('SCALING', conf_scaling.columns.tolist())
    confounds.set_group('TIMEPOINTS', conf_time_points.columns.tolist())
    
    # Add groups of continuous variable names for easy access
    confounds.set_group('AGE', conf_age.columns.tolist())
    confounds.set_group('HEAD_SIZE', conf_head_size.columns.tolist())
    confounds.set_group('TE', conf_te.columns.tolist())
    confounds.set_group('STRUCT_MOTION', conf_struct_head_motion.columns.tolist())
    confounds.set_group('DVARS', conf_dvars.columns.tolist())
    confounds.set_group('HEAD_MOTION', conf_head_motion.columns.tolist())
    confounds.set_group('HEAD_MOTION_ST', conf_head_motion_st.columns.tolist())
    confounds.set_group('TABLE_POS', conf_table_pos.columns.tolist())
    confounds.set_group('EDDY_QC', conf_eddy_qc.columns.tolist())
    
    # Add groups of age_sex variables for easy access
    confounds.set_group('AGE_SEX', conf_age_sex.columns.tolist())
    
    # Add group of subject level confounds
    confounds.set_group('SUBJECT', conf_age.columns.tolist() + \
                                   conf_sex.columns.tolist() + \
                                   conf_age_sex.columns.tolist() + \
                                   conf_head_size.columns.tolist())
    
    # Add group of acquisition related confounds
    confounds.set_group('ACQ', conf_site.columns.tolist() + \
                               conf_batch.columns.tolist() + \
                               conf_cmrr.columns.tolist() + \
                               conf_protocol.columns.tolist() + \
                               conf_service_pack.columns.tolist() + \
                               conf_scan_events.columns.tolist() + \
                               conf_flipped_swi.columns.tolist() + \
                               conf_fst2.columns.tolist() + \
                               conf_new_eddy.columns.tolist() + \
                               conf_scaling.columns.tolist() + \
                               conf_te.columns.tolist() + \
                               conf_time_points.columns.tolist())
    
    # Add group of motion related confounds
    confounds.set_group('MOTION', conf_struct_head_motion.columns.tolist() + \
                                  conf_dvars.columns.tolist() + \
                                  conf_head_motion.columns.tolist() + \
                                  conf_head_motion_st.columns.tolist())
                        
    # Add group of table position related confounds
    confounds.set_group('TABLE', conf_table_pos.columns.tolist() + \
                                 conf_eddy_qc.columns.tolist())

    
    # Delete previous dataframes
    del conf_site, categorical_IDPs, continuous_IDPs, conf_age_sex
    
    # Return the confounds
    return(confounds)