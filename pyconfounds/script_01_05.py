import os  
import time
import numpy as np  
import pandas as pd
from scipy.stats import f  
from scipy.stats import t 
from scipy.linalg import pinv, lstsq  

from nets.nets_demean import nets_demean
from nets.nets_pearson import nets_pearson
from nets.nets_load_match import nets_load_match

from preproc.switch_type import switch_type

from memmap.read_memmap_df import read_memmap_df
from memmap.addBlockToMmap import addBlockToMmap

from preproc.filter_columns_by_site import filter_columns_by_site

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
#  - method (int): Integer value of 1, 2 or 3, corresponding to the three 
#                  methods of computation which can be used.
#  - return_df (boolean): If true the result is returned as a pandas df,
#                         otherwise is is saved as a memory map (default 
#                         False).
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
def func_01_05_gen_nonlin_conf(data_dir, IDP_index, nonlinear_confounds, IDPs_deconf,
                               method=1, dtype=np.float64, p_fname=None, ve_fname=None,
                               return_df=False):
    
    # --------------------------------------------------------------------------------
    # Convert to appropriate datatype. If we have a filename for a memory mapped 
    # dataframe we want to read it in as a memory mapped df (after all it is already 
    # saved on disk so no extra memory usage there), otherwise if it is a pandas 
    # dataframe we leave it as it is (after all it is already in memory so we are not
    # increasing usage).
    # --------------------------------------------------------------------------------
    
    # If we only have filename
    if type(nonlinear_confounds) == str:
        
        # Convert input to memory mapped dataframes if it isn't already
        nonlinear_confounds = switch_type(nonlinear_confounds, out_type='MemoryMappedDF')
        
    # If we only have filename
    if type(IDPs_deconf) == str:
        
        # Convert input to memory mapped dataframes if it isn't already
        IDPs_deconf = switch_type(IDPs_deconf, out_type='MemoryMappedDF')

    # Get the subject ids
    sub_ids = IDPs_deconf.index
    
    # Read in the IDs for site
    site_ids = nets_load_match(os.path.join(data_dir, 'ID_SITE.txt'), sub_ids)
    
    # Get the unique site ids
    unique_site_ids = np.unique(site_ids)

    # If the method is 1-3
    if method <= 3:
        
        # Initialize indSite as a list to hold the indices
        inds_per_site = []
        
        # Loop over each value in site ids
        for site_id in unique_site_ids:
        
            # If we have a pandas dataframe
            if type(IDPs_deconf) == pd.core.frame.DataFrame:
                IDPs_values = IDPs_deconf.iloc[:, IDP_index].values.flatten()
            else:
                IDPs_values = IDPs_deconf[:, IDP_index].values.flatten()
        
            # Find the non-nan indices for this site
            indices = np.where(~np.isnan(IDPs_values) & (site_ids == site_id).all(axis=1))[0]
        
            # Append the found indices to the indSite list
            inds_per_site.append(indices)
        
        # Delete the indices
        del indices
        
        # Get the number of nonlinear confounds
        num_conf_nonlin = nonlinear_confounds.shape[1]
        
        # Get the number of IDPs
        num_IDPs = IDPs_deconf.shape[1]
        
        # Initialise empty arrays for p values
        p_df = pd.DataFrame(np.zeros((1,num_conf_nonlin))*np.NaN,columns=nonlinear_confounds.columns)
        
        # Initialise empty arrays for explained variances
        ve_df = pd.DataFrame(np.zeros((1,num_conf_nonlin))*np.NaN,columns=nonlinear_confounds.columns)
        
        # Get IDP
        if type(IDPs_deconf) == pd.core.frame.DataFrame:
            IDP = IDPs_deconf.iloc[:, IDP_index].values
        else:
            IDP = IDPs_deconf[:, IDP_index].values
        
        # If coincident we can speed things up by considering multiple columns at once
        for site_no in (unique_site_ids + 1):
        
            # Get the columns of nonlinear_confounds for this site
            nonlinear_confounds_site = filter_columns_by_site(nonlinear_confounds, site_no)
        
            # Check if we have enough values to perform the comparison
            if len(inds_per_site[site_no-1])!=0:
                
                # Subset to just this site (remembering zero indexing)
                nonlinear_confounds_site = nonlinear_confounds_site.iloc[inds_per_site[site_no-1],:]
                print(nonlinear_confounds_site.shape)
        
                # --------------------------------------------------------
                # Get X,Y and predicted Y
                # --------------------------------------------------------
                t1 = time.time()
                # Demean the confound data for the current site and nonlinear confound
                X = nets_demean(nonlinear_confounds_site).values
            
                # Get the IDP
                Y = IDP[inds_per_site[site_no-1]]
            
                # Get predicted Y = Xbeta (note this is being done seperately for each
                # column so we are not doing the usual inv(X.T @ X) @ X.T @ Y
                pred_Y = ((np.sum(X*Y,axis=0)/np.sum(X*X,axis=0))*X)
                print(pred_Y.shape)
            
                # Compute the residuals
                resids = Y - pred_Y
                
                # First method
                if method==1:
                    
                    # --------------------------------------------------------
                    # Variance explained version 1
                    # --------------------------------------------------------
                    # Get variance explained by pred_Y
                    ve = 100*((np.std(pred_Y,axis=0)/np.std(Y,axis=0))**2)
                
                    # --------------------------------------------------------
                    # P Version 1
                    # --------------------------------------------------------
                
                    # Compute the sum of squares for the effect
                    SSeffect = np.linalg.norm(pred_Y - np.mean(pred_Y),axis=0)**2  
                    
                    # Compute the sum of squares for the error
                    SSerror = np.linalg.norm(resids - np.mean(resids),axis=0)**2  
                    
                    # Degrees of freedom for the effect should be one as we are only 
                    # regressing the one column, unless a column has no observations
                    # at all
                    df = 1*np.any(np.abs(X)>1e-8,axis=0)
                    
                    # Compute the degrees of freedom for the error
                    dferror = len(Y) - df  
                    
                    # Compute the F-statistic
                    F = (SSeffect / df) / (SSerror / dferror)  
                    
                    # Compute p[i] using the F-distribution
                    p = 1 - f.cdf(F, df, dferror)  
        
                # Second method
                if method==2:
                    
                    # --------------------------------------------------------
                    # Variance explained version 2
                    # --------------------------------------------------------
                    
                    # Construct new design matrix
                    XplusIntercept = np.ones((X.shape[1], X.shape[0], 2))
                    XplusIntercept[:,:,1] = X.T[:]
        
                    # Perform OLS regression
                    U, D, Vt = np.linalg.svd(XplusIntercept, full_matrices=False)
                
                    # Get the rank of the matrix
                    rank = np.sum(D > 1e-10,axis=1)
                
                    # Rank reduce U, D and Vt
                    for i, rank_current in enumerate(rank):
                        U[i,:, rank_current:]=0 
                        Vt[i,rank_current:,:]=0
                        D[i,rank_current:]=0
                
                    # Get betahat
                    beta = (Vt.transpose((0,2,1))/D.reshape(*D.shape,1)) @ (U.transpose((0,2,1)) @ Y)
                
                    # Get residuals
                    resids = Y - XplusIntercept @ beta
                
                    # Get sigma^2 estimator
                    sigma2 = np.sum(resids**2,axis=1)/Y.shape[0]
                
                    # Contrast for beta2
                    L = np.array([0,1]).reshape((1,2,1))
                    
                    # Contrast variance
                    invDVtL = Vt/D.reshape(*D.shape,1) @ L 
                    varLtBeta = np.sqrt(sigma2.reshape(*sigma2.shape,1)*(invDVtL.transpose((0,2,1)) @ invDVtL))
                
                    # T statistic for contrast
                    T = L.transpose((0,2,1)) @ beta / varLtBeta
                            
                    # Second version of variance explained
                    ve = 100*(1-(np.std(resids,axis=1)**2/np.std(Y,axis=0)**2)).flatten()
                
                    # --------------------------------------------------------
                    # P-value version 2
                    # --------------------------------------------------------
                
                    # P value
                    p = 2*t.sf(np.abs(T.flatten()), dferror)
        
                # Third method
                if method==3:
                    
                    # --------------------------------------------------------
                    # P-value version 3
                    # --------------------------------------------------------
                
                    # Number of elements
                    n = X.shape[0]
                    
                    # Compute numerator
                    numerator = np.sum(X*Y,axis=0) - n*np.mean(X,axis=0)*np.mean(Y,axis=0)
                    
                    # Compute denominator
                    denom_X = np.sqrt(np.linalg.norm(X,axis=0)**2 - n*np.mean(X,axis=0)**2)
                    denom_Y = np.sqrt(np.linalg.norm(Y,axis=0)**2 - n*np.mean(Y,axis=0)**2)
                    
                    # Compute coefficient
                    R = numerator/(denom_X*denom_Y)
                     
                    # Get T statistic
                    T = R*np.sqrt((n-2)/(1-R**2))
                    
                    # Assuming 't' is your t-statistic and 'n' is sample size
                    p = 2*t.sf(np.abs(T), n-2)
                
                    # --------------------------------------------------------
                    # Variance explained version 3
                    # --------------------------------------------------------
                
                    # Compute version 3 of variance explained
                    ve = 100*R**2
        
                # Save p values and variance explained
                ve_df[[*nonlinear_confounds_site.columns]] = ve
                p_df[[*nonlinear_confounds_site.columns]] = p

    # Otherwise if the method is 4 we can broadcast over IDPs
    if method == 4:

        # Check the type of IDP index
        if type(IDP_index) in (np.int64, np.int32, np.int16, 'int64', 'int32', 'int16'):
            IDP_indices = [IDP_index]
        else:
            IDP_indices = IDP_index
        
        # Initialize indSite as a list to hold the indices
        inds_per_site = []
        
        # Loop over each value in site ids
        for site_id in unique_site_ids:
        
            # Find the indices for this site
            indices = np.where((site_ids == site_id).all(axis=1))[0]
        
            # Append the found indices to the indSite list
            inds_per_site.append(indices)
        
        # Delete the indices
        del indices
        
        # Get the number of nonlinear confounds
        num_conf_nonlin = nonlinear_confounds.shape[1]
        
        # Get the number of IDPs
        num_IDPs = IDPs_deconf.shape[1]
        
        # Get the number of IDPs in the block
        num_IDPs_block = len(IDP_indices)
        
        # Initialise empty arrays for p values
        p_df = pd.DataFrame(np.zeros((num_IDPs_block,num_conf_nonlin))*np.NaN,columns=nonlinear_confounds.columns)
        
        # Initialise empty arrays for explained variances
        ve_df = pd.DataFrame(np.zeros((num_IDPs_block,num_conf_nonlin))*np.NaN,columns=nonlinear_confounds.columns)
        
        # Get IDP
        if type(IDPs_deconf) == pd.core.frame.DataFrame:
            IDP_block = IDPs_deconf.iloc[:, IDP_indices].values
        else:
            IDP_block = IDPs_deconf[:, IDP_indices].values
        
        # If coincident we can speed things up by considering multiple columns at once
        for site_no in (unique_site_ids + 1):
            
            # Get the columns of nonlinear_confounds for this site
            nonlinear_confounds_site = filter_columns_by_site(nonlinear_confounds, site_no)
        
            # Check if we have enough values to perform the comparison
            if (len(inds_per_site[site_no-1])!=0) & (nonlinear_confounds_site.shape[1]!=0):
                
                # Subset to just this site (remembering zero indexing)
                nonlinear_confounds_site = nonlinear_confounds_site.iloc[inds_per_site[site_no-1],:]
        
                # Demean the confound data for the current site and nonlinear confound
                X = nets_demean(nonlinear_confounds_site).values # MARKER: Needs demean adjustment after y nans
        
                # Number of confounds for site
                num_conf_site = nonlinear_confounds_site.shape[1]
                
                # --------------------------------------------------------
                # Get X,Y and predicted Y
                # --------------------------------------------------------
        
                # Get the IDP
                Y = IDP_block[inds_per_site[site_no-1],:]
        
                # Get zerod version
                Y_with_zeros = np.array(Y)
                Y_with_zeros[np.isnan(Y)]=0 # MARKER: Needs demean adjustment
        
                # Compute Y'Y. Here each column is treated sepeately so Y is (n x 1)
                # and Y'Y is a single value for each column
                YtY = np.einsum('ij,ij->j',Y_with_zeros, Y_with_zeros).reshape(1,num_IDPs_block)
        
                # Compute X'Y for each X and Y individually (i.e. for each column
                # of X and each column of Y we do the (1 x n) by (n x 1) matrix
                # multiplication to get a single value
                XtY = np.zeros((num_conf_site, num_IDPs_block))
        
                # Compute X'X. There is no easy way to do this in a broadcasted way
                # as we have to construct a different X for each pattern of NaN values
                # in Y.
                XtX = np.zeros((num_conf_site, num_IDPs_block))
        
                # Block storing number of observations for each column
                n_per_col = np.zeros(IDP_block.shape[1])
        
                # Loop through y columns (the nan removal we cannot broadcast)
                for IDP_no in range(IDP_block.shape[1]):
        
                    # Get current Y
                    Y_current = Y[:,IDP_no:(IDP_no+1)]
                    
                    # Find the non-nan indices for this site
                    indices = np.where(~np.isnan(Y_current))[0]
        
                    # Subset X and Y
                    X_current = X[indices,:]
                    Y_current = Y_current[indices,:]
        
                    # Save n
                    n_per_col[IDP_no] = Y_current.shape[0]
        
                    # Demean
                    X_current = X_current - np.mean(X_current, axis=0)
        
                    # Compute XtX current
                    XtX[:,IDP_no] = np.einsum('ij,ij->j', X_current, X_current)
        
                    # Compute XtY current
                    XtY[:,IDP_no] = np.einsum('ij,ik->j', X_current, Y_current)
        
                # Get betahat
                betahat = XtY/XtX
        
                # Get variance explained
                ve = 100*(XtY**2)/YtY/XtX
        
                # Get degrees of freedom
                df = 1*np.any(np.abs(X)>1e-8,axis=0)
        
                # Get error degrees of freedom
                dferror = n_per_col.reshape((1, num_IDPs_block)) - df.reshape((num_conf_site,1))
        
                # F stat
                F = ((XtY**2)/YtY/XtX)/(df.reshape((num_conf_site,1))/dferror)
            
                # Compute p[i] using the F-distribution
                p = 1 - f.cdf(F, df.reshape((num_conf_site,1)), dferror)
        
                # Save p values and variance explained
                ve_df[[*nonlinear_confounds_site.columns]] = ve.T
                p_df[[*nonlinear_confounds_site.columns]] = p.T

    # Convert back to numpy
    ve = ve_df.values.flatten()
    p = p_df.values.flatten()
    
    # Check if we are returning the result
    if not return_df:
        
        # Get the memmap filenames for p values
        if p_fname is None:
            p_fname = os.path.join(os.getcwd(),'temp_mmap', 'p.npy')
            
        # Get the memmap filenames for p values
        if ve_fname is None:
            ve_fname = os.path.join(os.getcwd(),'temp_mmap', 've.npy')
        
        # Check the type of IDP index
        if type(IDP_index) in (np.int64, np.int32, np.int16, 'int64', 'int32', 'int16'):
            
            # Indices for where to add to memmap
            indices = np.ix_([IDP_index],np.arange(num_conf_nonlin))

        # Else we have a list of indices
        else:
        
            # Indices for where to add to memmap
            indices = np.ix_(IDP_indices,np.arange(num_conf_nonlin))
            
        # Add p values to memory maps
        addBlockToMmap(p_fname, p, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
        
        # Add explained variance values to memory maps
        addBlockToMmap(ve_fname, ve, indices,(num_IDPs, num_conf_nonlin),dtype=np.float64)
    
    else:
    
        # Return ve and p
        return(ve,p)