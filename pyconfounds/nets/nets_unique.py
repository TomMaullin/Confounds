import numpy as np

# ==========================================================================
#
# The below function is identical to the source code for numpys _unique1d
# (https://github.com/numpy/numpy/blob/main/numpy/lib/_arraysetops_impl.py#L336),
# with one exception; it also allows the return of the permutation from
# argsort. This is useful as it cuts computation time in half when both
# argsort and unique must be run on the same array.
#
# --------------------------------------------------------------------------
#
# It takes as inputs:
#
#  - ar : array_like
#        Input array. Unless `axis` is specified, this will be flattened if it
#        is not already 1-D.
#  - return_index : bool, optional
#        If True, also return the indices of `ar` (along the specified axis,
#        if provided, or in the flattened array) that result in the unique array.
#  - return_inverse : bool, optional
#        If True, also return the indices of the unique array (for the specified
#        axis, if provided) that can be used to reconstruct `ar`.
#  - return_counts : bool, optional
#        If True, also return the number of times each unique item appears
#        in `ar`.
#  - return_perm* : bool, optional
#        If True, also return the permutation obtained by running argsort on
#        `ar`.
#            
# --------------------------------------------------------------------------
#
# Returns
#
#  - unique : ndarray
#       The sorted unique values.
#  - indices : ndarray, optional
#       The indices of the first occurrences of the unique values in the
#       original array. Only provided if `return_index` is True.
#  - inverse : ndarray, optional
#       The indices to reconstruct the original array from the
#       unique array. Only provided if `return_inverse` is True.
#  - counts : ndarray, optional
#       The number of times each of the unique values comes up in the
#       original array. Only provided if `return_counts` is True.
#  - perm* : ndarray, optional
#       The permutation obtained by running argsort on the original array. 
#       Only provided if `return_perm` is True.
#
# *The addition of these arguments/outputs is the only difference between
#  this function and that used for numpy's unique.
# ==========================================================================
def nets_unique(ar, return_index=False, return_inverse=False,
                    return_counts=False, return_perm=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    # Check for optional indices
    optional_indices = return_index or return_inverse

    # If we have optional indices
    if optional_indices:

        # perform argsort with quicksort if we need the index,
        # mergesort otherwise
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]

    # Otherwise we dont need index
    else:

        # Run internal sort
        ar.sort()
        aux = ar

    # Get mask for returning outputs
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    # Initial return values
    ret = (aux[mask],)

    # Add index if returning
    if return_index:
        ret += (perm[mask],)

    # Add inverse if returning
    if return_inverse:

        # Get inverse using permutation
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask

        # Add it to output
        ret += (inv_idx,)

    # If we are returning value counts 
    if return_counts:

        # Compute value counts
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)

    # If we are returning the permutation
    if return_perm:

        # Add it to the outputs
        ret += (perm,)

    # Return results
    return ret