SynthPop
========

.. image:: https://travis-ci.org/UDST/synthpop.svg?branch=master
   :alt: Build Status
   :target: https://travis-ci.org/UDST/synthpop

.. image:: https://coveralls.io/repos/UDST/synthpop/badge.svg?branch=master
   :alt: Test Coverage
   :target: https://coveralls.io/r/UDST/synthpop?branch=master

SynthPop is a reimplementation of `PopGen`_ using the modern scientific Python
stack, with a focus on performance and code reusability.

The SynthPop code is a completely new implementation of the algorithms
described in this reference, and the paper as well as this repository should be
cited if you use SynthPop:

Ye, Xin, Karthik Konduri, Ram Pendyala, Bhargava Sana and Paul Waddell. A Methodology to Match Distributions of Both Households and Person Attributes in the Generation of Synthetic Populations.  Transportation Research Board 88th Annual Meeting Compendium of Papers DVD. January 11-15, 2009

The paper is available here:
http://www.scag.ca.gov/Documents/PopulationSynthesizerPaper_TRB.pdf

.. _PopGen: http://urbanmodel.asu.edu/popgen.html

# Installation

```
virtualenv venv --python=python3.7
source venv/bin/activate
pip install -r requierements.txt
cd synthpop/
python setup.py develop
```
To run `Synthpop` you need a Census API that you can get one from [here](https://api.census.gov/data/key_signup.html). After you get and validate the API key you can add it as an enviromental variable to your environment as by adding to `/venv/bin/activate` the following line: 
`export CENSUS='yourApiKey'`
