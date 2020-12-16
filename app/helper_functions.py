import os

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