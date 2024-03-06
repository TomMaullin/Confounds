# ------------------------------------------------------------
# Filters columns based on the specified site number.
#
# Parameters:
# - df: pandas.DataFrame to filter.
# - site_number: The site number to filter columns by.
#
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