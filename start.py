import subprocess
import sys

PACKAGE = ["flask","numpy","tensorflow==2.15"]

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while installing {package}: {e}")

for i in PACKAGE :
    install(i)


print("SUCCESSFULL INSTALLED PACKAGE")