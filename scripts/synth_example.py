from synthpop.recipes.starter import Starter
from synthpop.synthesizer import synthesize_all, enable_logging 
import os

def synthesize_county(county):
    starter = Starter(os.environ["CENSUS"], "CA", county)
    synthetic_population = synthesize_all(starter)
    return synthetic_population
    
synthesize_county('Solano County')
