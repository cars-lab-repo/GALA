## setup.py

from setuptools import setup, find_packages

setup(
    name='gala-lipstick',
    version='1.0.0',
    description='GNN-based Oracle-Less Logic Locking Attacks',
    author='Yeganeh Aghamohammadi',
    author_email='yeganeh@ucsb.edu',
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.0',
        'torch-geometric>=2.1.0',
        'networkx>=2.8',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pyyaml>=6.0',
        'scikit-learn>=1.0.0',
        'tqdm>=4.62.0',
    ],
    python_requires='>=3.8',
)
