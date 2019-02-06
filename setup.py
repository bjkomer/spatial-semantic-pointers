from setuptools import find_packages, setup

setup(
    name="spatial_semantic_pointers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'nengo',
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
    ]
)
