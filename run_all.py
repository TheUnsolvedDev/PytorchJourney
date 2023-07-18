import sys
import os

if __name__ == '__main__':
    files = [i for i in os.listdir(
        './') if i.endswith('.py') and i != 'run_all.py']

    for i in files:
        os.system('python3 '+i)
