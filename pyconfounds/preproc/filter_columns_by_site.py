from memmap.MemoryMappedDF import MemoryMappedDF

# ------------------------------------------------------------
# Filters columns based on the specified site number.
#
# Parameters:
# - df: pandas.DataFrame or Memory Mapped Dataframe to filter.
# - site_number: The site number to filter columns by.
# - return_df: If true the filtered dataframe is returned, else
#              just the list of columns
#
# Returns:
# - A DataFrame with only the columns that match the specified
#   site number, or just the column headers for the dataframe.
# ------------------------------------------------------------
def filter_columns_by_site(df, site_number, return_df=True):
    
    # Construct the site string to match in column names
    site_str = f"Site_{site_number}"
    
    # Filter columns that contain the site string
    filtered_columns = [col for col in df.columns if site_str in col]
    
    # Return the DataFrame with only the filtered columns
    if return_df:

        # Check the type of the dataframe to return
        if type(df)==MemoryMappedDF:
            return(df[:,filtered_columns])
        else:
            return(df[filtered_columns])

    # Otherwise return just column names
    else:
        return(filtered_columns)