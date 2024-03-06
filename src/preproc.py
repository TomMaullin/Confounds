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

# ------------------------------------------------------------
# Filters columns based on the specified site number.

# Parameters:
# - df: pandas.DataFrame to filter.
# - site_number: The site number to filter columns by.

# Returns:
# - A DataFrame with only the columns that match the specified
#   site number.
# ------------------------------------------------------------
def filter_columns_by_site(df, site_number):
    
    # Construct the site string to match in column names
    site_str = f"_Site_{site_number}"
    
    # Filter columns that contain the site string
    filtered_columns = [col for col in df.columns if site_str in col]
    
    # Return the DataFrame with only the filtered columns
    return df[filtered_columns]