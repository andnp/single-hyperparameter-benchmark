from setuptools import setup, find_packages

setup(
    name='shb',
    url='https://github.com/andnp/single-hyperparameter-benchmark.git',
    author='Andy Patterson',
    author_email='ap3@ualberta.ca',
    packages=find_packages(exclude=['tests*', 'paper*']),
    install_requires=[
        "PyExpUtils>=2.4",
    ],
    version=0.0,
    license='MIT',
    description='todo',
    long_description='todo',
)
