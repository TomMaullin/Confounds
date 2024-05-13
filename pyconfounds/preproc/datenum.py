import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# The below function takes in a year month and day and returns
# the `datenum`, replicating the behaviour of matlabs datenum
# function
# ------------------------------------------------------------
def datenum(year, month, day):
    
    # Get the date
    target_date = datetime(year, month, day)
    
    # Get the `first` date in datetime
    start_date = datetime(1, 1, 1)
    
    # Work out the difference
    delta = target_date - start_date
    
    # Adding 1 because MATLAB starts from January 0, 0000 but 
    # python starts from January 1, 0001
    return(delta.days + 1)