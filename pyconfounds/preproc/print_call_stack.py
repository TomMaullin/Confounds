import traceback

# -----------------------------------------------
# Helper function for debugging python errors.
# -----------------------------------------------
def print_call_stack():
    stack = traceback.format_stack()
    print("Call stack:")
    for line in stack:
        print(line.strip())
