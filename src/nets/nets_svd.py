import numpy as np

# ==========================================================================
#
# Performs an SVD on x. 
#
# Warning: This method composes x'x which has a square condition number when
#          compared to that of x. As a result, this will be less numerically
#          stable than performing an svd on x directly. However, if x is (m
#          by n) and m<<n or n<<m this will be a lot faster.
# 
# --------------------------------------------------------------------------
#
# Parameters:
#  - x (np.array): data to be decomposed.
#  - reorder (boolean): If we need the singular values in the same order as
#                       numpy output, setting this to true will do that.
#  - tol (float): tolerance level. Singular values of size less than this 
#                 are ignored.
#   
# --------------------------------------------------------------------------
#
# Returns:
#  - U (np.array): Left singular vectors of SVD.
#  - D (np.array): Singular values of SVD.
#  - Vt (np.array): Right singular vectors of SVD.
#     
# ==========================================================================
def nets_svd(x, reorder=True, tol=1e-10):

    # Get dimensions of x
    m, n = x.shape

    # Check which dimension of x is larger
    if m < n:

        # Transpose
        x = x.T

        # Record that we are looking at the transpose
        transposed = True

    # Otherwise we didnt transpose
    else:

        # Record that we didn't transpose
        transposed = False

    # Get X'X
    xtx = x.T @ x

    # Get eigendecomposition of x
    s, v = np.linalg.eigh(xtx)

    # Square root s
    s = np.sqrt(np.abs(s))

    # Reorder u, v and s appropriately
    if reorder:
        
        v = v[:,::-1]
        s = s[::-1]

    # Mask out zero values
    v = v[:,s>tol]
    s = s[s>tol]
    
    # Compute u
    u = x @ (v/s)
    
    # If we didn't transpose
    if not transposed:

        # Return u, s, vt
        return(u,s,v.T)

    # Otherwise 
    else:
        
        # Return ut, s, v
        return(v,s,u.T)    



