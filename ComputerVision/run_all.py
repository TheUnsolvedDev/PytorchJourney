import sys
import os
import glob


def run_all():
    search_pattern = "*/**/*.py"
    excluded_file = "run_all.py"
    files = glob.glob(search_pattern,recursive=True)

    model_files = [file for file in files if file != excluded_file]
    model_files.sort()
    
    for model_file in model_files:
        os.system('python3 {}'.format(model_file))


def print_all(move = False):
    search_pattern = "*/**/*.py"
    excluded_file = "run_all.py"
    files = glob.glob(search_pattern,recursive=True)

    model_files = [file for file in files if file != excluded_file]
    model_files.sort()
    print("## Models\n")
    for file in model_files:
        model_name = file[:-3] 
        model_name = model_name.replace("_", " ").title()
        print("- [x] {} - `{}`".format(model_name, file))

if __name__ == '__main__':
    run_all()
    print_all(move = True)
    # run_all()
