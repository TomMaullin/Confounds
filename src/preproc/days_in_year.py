import numpy as np

# ------------------------------------------------------------
# The below gives the number of days for each year in a given
# array of years.
# ------------------------------------------------------------
def days_in_year(years):
    
    # Work out the leap years
    leap_years = ((years % 4) == 0) & ((years % 100) != 0)

    # Set days to an array of zeros
    days = np.zeros(years.shape)
    
    # Set leap years and non-leap years
    days[leap_years] = 366
    days[~leap_years] = 365

    # Return result
    return(days)