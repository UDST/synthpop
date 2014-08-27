from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(
    name='popgen',
    version='0.1dev',
    description='Population Synthesis',
    author='Synthicity',
    author_email='pwaddell@synthicity.com',
    license='AGPL',
    url='https://github.com/synthicity/popgen',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU Affero General Public License v3'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'census>=0.6',
        'numexpr>=2.3.1',
        'numpy>=1.8.0',
        'pandas>=0.13.1',
        'scipy>=0.13.3'
    ]
)
