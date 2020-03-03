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

==============
How to run it
==============
0. You need to have installed Python >= 3.6
1. You have forked repo
2. Open terminal
3. If it is first time, install all dependencies via running ``python3 setup.py install``
4. You need to have Census API key in order to query the date. Sign up using the following link: https://api.census.gov/data/key_signup.html
5. Run ``CENSUS=#API_KEY# python3 demos/synthesize.py TX "Travis County"`` to start generating population for the state Texas (TX) and state Travis County. If you want to generate the data on census tract level, you can pass extra parameteres, an example: ``python3 demos/synthesize.py "TX" "Travis County" 48 453 001804 1``, where 48 - FIPS code of Texas, 453 - FIPS code of Travis county, 001804 - tract code, 1 - block group
