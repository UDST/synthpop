from synthpop.recipes.starter2 import Starter
from synthpop.synthesizer import synthesize_all, enable_logging 
import os

def synthesize_county(county):
    starter = Starter(os.environ["CENSUS"], "CO", county)
    synthetic_population = synthesize_all(starter)
    return synthetic_population
    
synthesize_county('Gilpin County')
