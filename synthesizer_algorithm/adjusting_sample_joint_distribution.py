import os
import pandas as pd
import numpy as np
import synthesizer_algorithm.drawing_households
import synthesizer_algorithm.pseudo_sparse_matrix
import time
from scipy import sparse

def create_joint_dist():
    housing_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','frequency','hhuniqueid'])
    person_synthetic_data = pd.DataFrame(columns=['state','county','tract','bg','hhid','serialno','pnum','frequency','personuniqueid'])

