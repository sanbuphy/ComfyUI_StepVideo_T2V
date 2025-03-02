import os
import sys
import subprocess
import pkg_resources

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read requirements.txt file
requirements_path = os.path.join(script_dir, "requirements.txt")
with open(requirements_path, 'r') as f:
    requirements = [line.strip() for line in f if line.strip()]

# Check and install missing dependencies
print("Checking and installing required dependencies...")
for requirement in requirements:
    try:
        pkg_name = requirement.split('>=')[0].split('==')[0]
        pkg_resources.get_distribution(pkg_name)
        print(f"✓ {pkg_name} is already installed")
    except pkg_resources.DistributionNotFound:
        print(f"Installing {requirement}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        print(f"✓ {requirement} installation completed")

print("\nAll dependencies have been installed!")
print("When running the node for the first time, the StepVideo T2V model (approx. 24GB) will be automatically downloaded.")
print("Please ensure you have sufficient disk space and network connectivity.") 