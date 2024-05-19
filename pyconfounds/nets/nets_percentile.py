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