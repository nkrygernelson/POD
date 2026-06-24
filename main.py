import subprocess
import sys

def main():
    extra = sys.argv[1:]   # pass CLI overrides through to both
    subprocess.run([sys.executable, "train.py", *extra], check=True)
    subprocess.run([sys.executable, "generation.py", *extra], check=True)

if __name__ == "__main__":
    main()