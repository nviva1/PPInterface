from setuptools import setup, find_packages

# Read dependencies from requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='your-package-name',
    version='0.1.0',
    description='A description of your package',
    long_description=open('README.md').read(),  # Optional: use README as long description
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
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
)
