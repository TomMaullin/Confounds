import numpy as np
from scipy.stats import t

# ----------------------------------------------------------------
#
# This was `my_nancorr` in the matlab version of the code. It 
# computes the pearsn coefficient between x and y.
#
# ----------------------------------------------------------------
#
# It takes as inputs:
# - x,y: numpy arrays, vectors to find correlation between.
#
# ----------------------------------------------------------------
#
# And returns:
# - R: The Pearson correlation coefficient.
#
# ----------------------------------------------------------------
def nets_pearson(x,y):
    
    # Number of elements
    n = x.shape[0]
    
    # Compute numerator
    numerator = np.sum(x*y) - n*np.mean(x)*np.mean(y)
    
    # Compute denominator
    denom_x = np.sqrt(np.linalg.norm(x)**2 - n*np.mean(x)**2)
    denom_y = np.sqrt(np.linalg.norm(y)**2 - n*np.mean(y)**2)
    
    # Compute coefficient
    R = numerator/(denom_x*denom_y)
     
    # Get T statistic
    T = R*np.sqrt((n-2)/(1-R**2))
    
    # Assuming 't' is your t-statistic and 'n' is sample size
    p = 2*t.sf(np.abs(T), n-2)
    
    # Return pearsons coefficient
    return(R,p)
    