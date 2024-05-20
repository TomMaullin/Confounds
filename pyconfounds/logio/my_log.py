import os

# ------------------------------------------------------------------------------
#
# The below function writes a message to an HTML log file with a header, basic
# formatting, and styling.
#
# ------------------------------------------------------------------------------
#
# It takes as inputs:
#
#     message (str): The message to be written.
#     mode (str, optional): The mode to open the file with. 'a' for append, 
#                           'r' for replace current line. Defaults to 'a'.
#     filename (str, optional): The name of the HTML file. Defaults to None 
#                               (no output).
#
# ------------------------------------------------------------------------------

def my_log(message, mode='a',filename=None):
    
    # Check if file is in use
    fileLocked = True
    while fileLocked:
        try:
            # Create lock file, so other jobs know we are writing to this file
            f = os.open(filename + ".lock", os.O_CREAT|os.O_EXCL|os.O_RDWR)
            fileLocked = False
        except FileExistsError:
            fileLocked = True
            
    if filename is not None:
        try:
            # Read file line
            with open(filename, 'r+', encoding='utf-8') as file:
                lines = [line for line in file]

            # Writing filelines
            with open(filename, 'w+', encoding='utf-8') as file:
                if not lines:  # If the file is empty, create the HTML structure
                    file.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Confounds Log</title>\n')
                    file.write('<style>\nbody { font-family: Arial, sans-serif; margin: 20px; background-color: #e6f0ff; }\n')
                    file.write('h1 { color: #333; position: sticky; top: 0; background-color: #e6f0ff; padding: 10px; }\n')
                    file.write('hr { border: none; border-top: 1px solid #ccc; margin: 10px 0; }\n</style>\n</head>\n<body>\n<h1>Confounds Log</h1>\n<hr>\n')
                    file.write('<p>' + message + '</p>\n')
                    file.write('</body>\n')
                    file.write('</html>')
                else:
                    if mode == 'a':
                        lines.append('</html>')
                        lines[-3] = '<p>' + message + '</p>\n'
                        lines[-2] = '</body>\n'
                    elif mode == 'r':
                        lines[-3] = '<p>' + message + '</p>\n'
                        lines[-2] = '</body>\n'
                        lines[-1] = '</html>'
                    file.writelines(lines)
        except FileNotFoundError:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Confounds Log</title>\n')
                file.write('<style>\nbody { font-family: Arial, sans-serif; margin: 20px; background-color: #e6f0ff; }\n')
                file.write('h1 { color: #333; position: sticky; top: 0; background-color: #e6f0ff; padding: 10px; }\n')
                file.write('hr { border: none; border-top: 1px solid #ccc; margin: 10px 0; }\n</style>\n</head>\n<body>\n<h1>Confounds Log</h1>\n<hr>\n')
                file.write('<p>' + message + '</p>\n')
                file.write('</body>\n')
                file.write('</html>')

    # Release the file lock
    os.remove(filename + ".lock")
    os.close(f)