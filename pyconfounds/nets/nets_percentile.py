import numpy as np

# ---------------------------------------------------------
#
# The matlab and python percentile functions have very
# slightly differing behaviours. To handle this we use the
# below code taken from this stack exchange post:
#
# https://stackoverflow.com/questions/58424704/output-produced-by-python-numpy-percentile-not-same-as-matlab-prctile
#
# ---------------------------------------------------------
# 
# It takes as inputs:
#
#  - x (np.array): The data to find the percentile from.
#  - p (float/numeric): The percentile to find, between 0
#                       and 100.
#
# ---------------------------------------------------------
#
# It returns:
#
# - float/numeric: The percentile p from the data x (matlab
#                  version).
#
# ---------------------------------------------------------
def nets_percentile(x, p):

    # Make sure we have any array
    p = np.asarray(p, dtype=float)

    # Get its length
    n = len(x)

    # Transform
    p = (p-50)*n/(n-1) + 50

    # Clip the range
    p = np.clip(p, 0, 100)

    # Return the result
    return(np.percentile(x, p))