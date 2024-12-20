from setuptools import setup, find_packages

setup(
    name='ForexMeanReversion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'sklearn'
    ],
    python_requires='>=3.8',
)