import sys
import os
import glob


def run_all():
    files = [i for i in os.listdir(
        './') if i.endswith('.py') and i != 'run_all.py']
    for i in files:
        os.system('python3 '+i)


def print_all():
    search_pattern = "*.py"
    excluded_file = "run_all.py"
    files = glob.glob(search_pattern)

    model_files = [file for file in files if file != excluded_file]
    model_files.sort()
    print("## Models\n")
    for file in model_files:
        model_name = file[:-3] 
        model_name = model_name.replace("_", " ").title()
        print("- [x] {} - `{}`".format(model_name, file))


if __name__ == '__main__':
    # run_all()
    print_all()
    run_all()
