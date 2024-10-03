from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import sys

#class PostInstallCommand(install):
#   """Post-installation for installation mode."""
#   
#   def run(self):
#       # First, run the normal install process
#       self.do_egg_install()       
#       install.run(self)
#
#       # Get the absolute path to the 'proteinmpnn_wrapper' folder
#       current_dir = os.path.abspath(os.path.dirname(__file__))
#       submodule_dir = os.path.join(current_dir, 'proteinmpnn_wrapper')
#
#       # Check if the 'proteinmpnn_wrapper' directory exists
#       if os.path.isdir(submodule_dir):
#           try:
#               # Change directory to 'proteinmpnn_wrapper'
#               os.chdir(submodule_dir)
#               
#               # Install the package in 'proteinmpnn_wrapper'
#               subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
#               print("Successfully installed the proteinmpnn_wrapper package.")
#           except subprocess.CalledProcessError as e:
#               print(f"Error during installation of proteinmpnn_wrapper: {e}")
#       else:
#           print(f"'proteinmpnn_wrapper' directory not found at {submodule_dir}. Skipping installation.")

# Helper function to read requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='PPInterface',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "biotite==0.40.0",
        "biopandas",
        "biopython",
        "omegaconf",
        'git+https://github.com/Croydon-Brixton/proteinmpnn_wrapper'
    ],
    #install_requires=read_requirements(),  # Automatically install packages from requirements.txt
    #cmdclass={#
    #    'install': PostInstallCommand,
    #},
)
