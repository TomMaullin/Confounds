def ascii_loading_bar(percentage):
    """
    Generate an ASCII loading bar based on the given percentage.

    Args:
        percentage (float): The percentage value (between 0 and 100).

    Returns:
        str: The ASCII loading bar string.
    """
    # Ensure the percentage is within the valid range
    percentage = max(0, min(100, percentage))

    # Calculate the number of filled and empty bars
    bar_length = 50  # Adjust this value to change the length of the loading bar
    filled_bars = int(bar_length * (percentage / 100))
    empty_bars = bar_length - filled_bars

    # Create the loading bar string
    bar = '[' + '&#9608;' * filled_bars + '&#9619;' * empty_bars + ']'

    # Format the percentage string
    percentage_str = f"{percentage:.1f}%"

    # Combine the loading bar and percentage
    loading_bar = f"{bar} {percentage_str}"

    return loading_bar