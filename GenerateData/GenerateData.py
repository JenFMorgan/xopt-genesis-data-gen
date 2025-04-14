#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from xopt import Evaluator
from xopt import VOCS
import math
from xopt.generators import get_generator 
from xopt import Xopt

#from xopt import AsynchronousXopt as Xopt
from Genesis_functions import *

from TaperFunctions import *


import json
import numpy as np
from pmd_beamphysics import ParticleGroup
from pathlib import Path
import os
from pathlib import Path
import hashlib
Genesis4.apply_taper = apply_taper


# In[10]:


YAML = """

max_evaluations: 7000

strict: false

dump_file: dump_cnsga.yml
evaluator:
    function: Genesis_functions.evaluate_genesis
    function_kwargs: {
    run_file_location: /sdf/home/j/jmorgan/lclsrepo_dontsharethese/lcls-lattice/genesis/version4/cu_hxr/,
    run_file: cu_hxr.in,
    Lattice_file: hxr.lat,
    workdir: /sdf/scratch/users/j/jmorgan,
    nproc: 30}
    vectorized: false

generator:
    name: cnsga
    output_path: temp
    population_size: 30

vocs:
    variables:
    
        taper_quadratic: [0, 0.2]
        alphax: [-2.5, 1]
        alphay: [1, 2.5]
        betax: [5, 30]
        betay: [5, 30]
        emitx: [1e-7, 1e-6]
        emity: [1e-7, 1e-6]
        chirp: [-20 , 0]  
        beam_length:  [15e-6, 18e-6]
        u_linear_stop : [10, 16]  
        
    constraints:
        energy: [GREATER_THAN, 100]
    objectives: 
        energy: MAXIMIZE
        bandwidth_FWHM: MAXIMIZE 

    observables: [bandwidth_FWHM, peak_energy, peak_fluence, peak_intensity]
"""


# In[12]:

X2= Xopt.from_yaml(YAML)

#X2.run()

from concurrent.futures import ProcessPoolExecutor

N_CPUS = 5 # Define number of CPUs

with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
    X2.evaluator.executor = executor
    X2.evaluator.max_workers = N_CPUS
    X2.run()



X2.generator.write_population("test.csv")



# In[ ]:




