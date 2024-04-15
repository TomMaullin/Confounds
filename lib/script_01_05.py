import os  
import numpy as np  
import pandas as pd
from scipy.stats import f  
from scipy.stats import t 
from scipy.linalg import pinv, lstsq  
from src.nets.nets_demean import nets_demean
from src.nets.nets_pearson import nets_pearson
from src.preproc.switch_type import switch_type
from src.memmap.read_memmap_df import read_memmap_df
from src.nets.nets_load_match import nets_load_match
from src.memmap.addBlockToMmap import addBlockToMmap

# =============================================================================
#
# This function computes various statistics and metrics related to non-linear
# confounds and image derived phenotypes (IDPs) for a given index (IDP_index).
#
# -----------------------------------------------------------------------------
# 
# It takes the following inputs:
#
#  - data_dir (string): The directory containing the data.
#  - IDP_index (int): The index of the IDP to be analyzed.
#  - nonlinear_confounds (str): Filename for memory mapped nonlinear confounds.
#  - IDPs_deconf (str): Filename for memory mapped deconfounded IDPs.
#
# -----------------------------------------------------------------------------
#
# It saves all p-value results to files named p*.npy and variance explained 
# values to ve*.npy.
#
# -----------------------------------------------------------------------------
#
# Developer's note: This file does not correspond to the bash script named
# script_01_05_gen_nonlin_conf.sh in the matlab repo. Instead, it corresponds
# to the matlab function func_01_05_gen_nonlin_conf.m.
#
# =============================================================================
def func_01_05_gen_nonlin_conf(data_dir, IDP_index, nonlinear_confounds, IDPs_deconf, dtype=np.float64):

    # --------------------------------------------------------------------------------
    # Convert to memory mapped df (if not already)
    # --------------------------------------------------------------------------------
        
    # Convert input to memory mapped dataframes if it isn't already
    nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF')
    IDPs_deconf = switch_type(IDPs_deconf, out_type='MemoryMappedDF')

    # Get the subject ids
    sub_ids = IDPs_deconf.index

    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)

    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)

    # Initialize indSite as a list to hold the indices
    inds_per_site = []

    # Loop over each value in site ids
    for site_id in unique_site_ids:

        # Find the non-nan indices for this site
        indices = np.where(~np.isnan(IDPs_deconf[:, IDP_index].values.flatten()) & (site_ids == site_id).all(axis=1))[0]

        # Append the found indices to the indSite list
        inds_per_site.append(indices)

    # Delete the indices
    del indices
    
    # Get the number of nonlinear confounds
    num_conf_nonlin = nonlinear_confounds.shape[1]
    
    # Get the number of IDPs
    num_IDPs = IDPs_deconf.shape[1]
    
    # Initialise empty arrays for p values
    p1 = np.zeros(num_conf_nonlin)
    p2 = np.zeros(num_conf_nonlin)
    p3 = np.zeros(num_conf_nonlin)
    
    # Initialise empty arrays for explained variances
    ve1 = np.zeros(num_conf_nonlin)
    ve2 = np.zeros(num_conf_nonlin)
    ve3 = np.zeros(num_conf_nonlin)
    
    # Loop through confounds
    for i in range(num_conf_nonlin):
        
        # Get the site of the non-linear confound
        site_no = int(nonlinear_confounds.columns[i].split('Site_')[1][0])
    
        # Check if we have enough values to perform the comparison
        if len(inds_per_site[site_no-1])!=0:
            
            # Get the nonlinear confound for this site
            nonlinear_confound = nonlinear_confounds[:,i].values
            
            # Subset to just this site (remembering zero indexing)
            nonlinear_confound = nonlinear_confound[inds_per_site[site_no-1]]
        
            # --------------------------------------------------------
            # Get X,Y and predicted Y
            # --------------------------------------------------------
        
            # Demean the confound data for the current site and nonlinear confound
            X = nets_demean(pd.DataFrame(nonlinear_confound)).values
        
            # Get Y
            Y = IDPs_deconf[:, IDP_index].values
            Y = Y[inds_per_site[site_no-1]]
        
            # Remove potential nans from X
            Y = Y[~np.isnan(X)]
            X = X[~np.isnan(X)]
        
            # Get predicted Y = Xbeta
            pred_Y = np.nansum(X*Y)/np.nansum(X**2)*X
        
            # Compute the residuals
            resids = Y - pred_Y
        
            # --------------------------------------------------------
            # Variance explained version 1
            # --------------------------------------------------------
            # Get variance explained by pred_Y
            ve1[i] = 100*((np.nanstd(pred_Y)/np.std(Y[~np.isnan(X)]))**2)
        
            # --------------------------------------------------------
            # P Version 1
            # --------------------------------------------------------
        
            # Compute the sum of squares for the effect
            SSeffect = np.linalg.norm(pred_Y - np.mean(pred_Y))**2  
        
            # Compute the sum of squares for the error
            SSerror = np.linalg.norm(resids - np.mean(resids))**2  
        
            # Compute the degrees of freedom for the effect
            df = np.linalg.matrix_rank(X) 
        
            # Compute the degrees of freedom for the error
            dferror = len(Y) - df  
        
            # Compute the F-statistic
            F = (SSeffect / df) / (SSerror / dferror)  
        
            # Compute p[i] using the F-distribution
            p1[i] = 1 - f.cdf(F, df, dferror)  
        
            # --------------------------------------------------------
            # Variance explained version 2
            # --------------------------------------------------------
        
            # Construct new design matrix
            XplusIntercept = np.ones((X.shape[0],2))
            XplusIntercept[:,1] = X[:]
        
            # Perform OLS regression
            U, D, Vt = np.linalg.svd(XplusIntercept, full_matrices=False)
        
            # Get the rank of the matrix
            rank = np.sum(D > 1e-10)
        
            # Rank reduce U, D and Vt
            U = U[:, :rank] 
            Vt = Vt[:rank,:]
            D = D[:rank]
        
            # Get betahat
            beta = (Vt.T/D) @ (U.T @ Y)
        
            # Get residuals
            resids = Y - XplusIntercept @ beta
        
            # Get sigma^2 estimator
            sigma2 = np.sum(resids**2)/Y.shape[0]
        
            # Contrast for beta2
            L = np.array([[0],[1]])
        
            # Contrast variance
            invDVtL = Vt/D @ L 
            varLtBeta = np.sqrt(sigma2*invDVtL.T @ invDVtL)
        
            # T statistic for contrast
            T = L.T @ beta / varLtBeta
        
            # Second version of variance explained
            ve2[i] = 100*(1-(np.std(resids)**2/np.std(Y)**2))
        
        
            # --------------------------------------------------------
            # P-value version 2
            # --------------------------------------------------------
        
            # P value
            p2[i] = 2*t.sf(np.abs(T), dferror)[0,0]
        
            # --------------------------------------------------------
            # P-value version 3
            # --------------------------------------------------------
        
            # Compute pearson coefficient
            R, p3[i] = nets_pearson(X,Y)
        
            # --------------------------------------------------------
            # Variance explained version 3
            # --------------------------------------------------------
        
            # Compute version 3 of variance explained
            ve3[i] = 100*R**2
    
        else:
    
            # Set values to NaN
            p1[i] = np.NaN
            p2[i] = np.NaN
            p3[i] = np.NaN
            ve1[i] = np.NaN
            ve2[i] = np.NaN
            ve3[i] = np.NaN
            
        
    # Get the memmap filenames for p values
    p1_mmap = os.path.join(os.getcwd(),'temp_mmap', 'p1.npy')
    p2_mmap = os.path.join(os.getcwd(),'temp_mmap', 'p2.npy')
    p3_mmap = os.path.join(os.getcwd(),'temp_mmap', 'p3.npy')
        
    # Get the memmap filenames for p values
    ve1_mmap = os.path.join(os.getcwd(),'temp_mmap', 've1.npy')
    ve2_mmap = os.path.join(os.getcwd(),'temp_mmap', 've2.npy')
    ve3_mmap = os.path.join(os.getcwd(),'temp_mmap', 've3.npy')
    
    # Indices for where to add to memmap
    indices = np.ix_([IDP_index],np.arange(num_conf_nonlin))
    
    # Add p values to memory maps
    addBlockToMmap(p1_mmap, p1, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    addBlockToMmap(p2_mmap, p2, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    addBlockToMmap(p3_mmap, p3, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    
    # Add explained variance values to memory maps
    addBlockToMmap(ve1_mmap, ve1, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    addBlockToMmap(ve2_mmap, ve2, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    addBlockToMmap(ve3_mmap, ve3, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)