import os
import numpy as np
import pandas as pd
from src.nets.nets_svd import nets_svd
from src.nets.nets_demean import nets_demean
from src.preproc.switch_type import switch_type
from src.memmap.MemoryMappedDF import MemoryMappedDF
from src.memmap.addBlockToMmap import addBlockToMmap

# Note columns must have same nan_patterns in y.
def nets_deconfound_single(y, conf, columns, mode='nets_svd', demean=True, dtype=np.float64, out_fname=None):
    
    # Switch type to save transfer costs 
    conf = switch_type(conf, out_type="pandas") # Only time all data is read in
    y = switch_type(y,out_type="MemoryMappedDF")
    
    # Get dimensions we are ouputting to
    out_dim = y.shape
    
    # Save original index and columns for outputting later
    y_index_original = y.index
    y_columns_original = y.columns
    
    # Get the y's we're interested in
    y_current = y[:,columns]
    
    # If we have subset the data we need to demean again
    if demean:
        
        # Demean y and conf
        y_current = nets_demean(y_current)
    

    # If we don't have nans recorded work them out
    non_nan_rows = ~y_current.isna().any(axis=1)
        
    # Subset y and conf to the appropriate rows
    y_current = y_current[non_nan_rows]
    conf_current = conf[non_nan_rows]

    # Save y index and columns
    y_index = y_current.index
    y_columns = y_current.columns
    
    # If we are demeaning
    if demean:
        
        # Demean conf_current
        conf_current = nets_demean(conf_current)
        
    # We don't want to work on views of the data as it will slow the computation
    conf_current = np.array(conf_current.values)
    y_current = np.array(y_current.values)
        
    # Check if we are using psuedo inverse
    if mode.lower() == 'pinv':
    
        # Regress conf out of y_current - we perform the pseudo inverse on
        # conf^T @ conf as we expect the number of columns to be much(!) less
        # than the number of rows and thus this ends up being more numerically
        # stable than trying to invert, or approximately invert, conf itself.
        betahat = np.linalg.pinv(conf_current.T @ conf_current) @ conf_current.T @ y_current
    
        # Set computational zeros to actual zeros
        betahat[np.abs(betahat) < 1e-10] = 0
    
        # Get deconfounding variable predicted values to regress out
        deconf_pred = pd.DataFrame(conf_current @ betahat)
        deconf_pred.index = y_index
        deconf_pred.columns = y_columns
    
    # Otherwise use svd
    elif mode.lower() == 'nets_svd':
        
        # Multiply the left-singular values which contribute to the rank of conf
        # by the corresponding singular values to rank reduce conf
        U, _, _ = nets_svd(conf_current, reorder=False)
        
        # Get deconfounding variable predicted values to regress out
        deconf_pred = pd.DataFrame(U @ (U.T @ y_current))
        deconf_pred.index = y_index
        deconf_pred.columns = y_columns
        
    # Otherwise use precomputed svd (in this case we assume we were handed U in the
    # place of conf)
    elif mode.lower() == 'svd_precompute':
        
        # Get deconfounding variable predicted values to regress out
        deconf_pred = pd.DataFrame(conf_current @ (conf_current.T @ y_current))
        deconf_pred.index = y_index
        deconf_pred.columns = y_columns
    
    # Otherwise use svd
    elif mode.lower() == 'svd':
        
        # Multiply the left-singular values which contribute to the rank of conf
        # by the corresponding singular values to rank reduce conf
        U, S, _ = np.linalg.svd(conf_current, full_matrices=False)
        
        # Get the rank of the matrix
        rank = np.sum(S > 1e-10)
        
        # Rank reduce U and reduce datatype as only need to multiply
        U = U[:, :rank]
        
        # Get deconfounding variable predicted values to regress out
        deconf_pred = pd.DataFrame(U @ (U.T @ y_current))
        deconf_pred.index = y_index
        deconf_pred.columns = y_columns
        
    else:
    
        # Perform qr decomposition
        Q, R = np.linalg.qr(conf_current)
        betahat = np.linalg.pinv(R) @ (Q.T @ y_current)
    
        # Set computational zeros to actual zeros
        betahat[np.abs(betahat) < 1e-10] = 0
    
        # Get deconfounding variable predicted values to regress out
        deconf_pred = pd.DataFrame(conf_current @ betahat)
        deconf_pred.index = y_index
        deconf_pred.columns = y_columns
        
    # Get deconfounded y
    y_deconf_current = pd.DataFrame(y_current, index=y_index, columns=y_columns) - deconf_pred
        
    # If we are demeaning, demean y
    if demean:
        y_deconf_current = nets_demean(y_deconf_current)
        
    # Update deconfounded y 
    y_deconf_current_with_nans = np.ones((len(y_index_original), 
                                          len(y_deconf_current.columns)))*np.NaN
        
    # Update with current values
    y_deconf_current_with_nans[non_nan_rows] = y_deconf_current.values
    
    # Make into a dataframe with correct index and rows
    y_deconf_current_with_nans = pd.DataFrame(y_deconf_current_with_nans,
                                              index=y_index_original,
                                              columns=y_deconf_current.columns)
    
    # Indices for where to add to memmap (note: everything is transposed as these 
    # files are **much** quicker to save row by row than column by column
    indices = np.ix_([list(y_columns_original).index(column) for column in columns],
                     np.arange(out_dim[0]))

    # Add the block to the memory map
    addBlockToMmap(out_fname,  y_deconf_current_with_nans.values.T, indices, (out_dim[1],out_dim[0]), dtype=dtype)

