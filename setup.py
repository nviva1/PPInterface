from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os


# Read dependencies from requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

#class PostInstallCommand(install):
#    """Post-installation for installing submodules."""
#    def run(self):
#        # Clone submodules
#        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
#        # Continue with the installation
#        install.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    
    def run(self):
        # First, run the normal install process
        install.run(self)
        
        # Get the absolute path to the 'proteinmpnn_wrapper' folder
        current_dir = os.path.abspath(os.path.dirname(__file__))
        submodule_dir = os.path.join(current_dir, 'proteinmpnn_wrapper')

        # Check if the 'proteinmpnn_wrapper' directory exists
        if os.path.isdir(submodule_dir):
            try:
                # Change directory to 'proteinmpnn_wrapper'
                os.chdir(submodule_dir)
                
                # Install the package in 'proteinmpnn_wrapper'
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
                print("Successfully installed the proteinmpnn_wrapper package.")
            except subprocess.CalledProcessError as e:
                print(f"Error during installation of proteinmpnn_wrapper: {e}")
        else:
            print(f"'proteinmpnn_wrapper' directory not found at {submodule_dir}. Skipping installation.")

setup(
    name='PPInterface',
    version='0.1.0',
    description='TBA',
    long_description=open('README.md').read(),  # Optional: use README as long description
    long_description_content_type='text/markdown',
    author='Nikita Ivanisenko',
    author_email='n.ivanisenko@gmail.com',
    url='https://github.com/username/your-repo',  # Replace with the actual repo URL
    packages=find_packages(),
    install_requires=read_requirements(),  # Load requirements from the file
    classifiers=[
        'Development Status :: 3 - Alpha',  # Adjust based on your project's status
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Specify the minimum Python version
    cmdclass={
        'install': PostInstallCommand,
        }
)


