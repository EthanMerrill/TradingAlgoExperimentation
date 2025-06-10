import os
import traceback

# if app/tmp doesn't exist, make it:
def ensure_dir(file_path):
    try:     
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created filepath: {file_path}")
        else:
            print(f"filepath {file_path} already exists")
        return
    except Exception as e:
        print(f"seems you might be running this in a docker instance, lets try making the file a different way. Error: {e}")
    try:
        os.system("mkdir "+file_path)
        return
    except Exception as e:
        print(f"second method of creating directory failed: {e}")

def log_traceback(ex):
    tb_lines = traceback.format_exception(ex.__class__, ex, ex.__traceback__)
    tb_text = ''.join(tb_lines)
    # I'll let you implement the ExceptionLogger class,
    # and the timestamping.
    #NEED TO MAKE AN EXCEPTION LOGGER OR USE THE LOGGING MODULE
    print(tb_text)

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))