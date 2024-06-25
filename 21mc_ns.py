import matplotlib.pyplot as plt
import numpy as np

from py21cmmc import analyse
from py21cmmc import mcmc
import py21cmmc as p21mc
import logging
print("Version of py21cmmc: ", p21mc.__version__)

core = p21mc.CoreLightConeModule( # All core modules are prefixed by Core* and end with *Module
    redshift = 5.5,              # Lower redshift of the lightcone
    max_redshift = 12.0,          # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params = dict(
        HII_DIM = 80,
        BOX_LEN = 200.0,
        PERTURB_ON_HIGH_RES = False,
        USE_INTERPOLATION_TABLES = False,
        N_THREADS=2),
    flag_options = dict(
        INHOMO_RECO = True,
        USE_TS_FLUCT = True,
    ),
    direc="_cache",
    regenerate=True,
    cache_mcmc=False,
    cache_dir="_cache",
    cache_ionize=False

) # For other available options, see the docstring.

# Now the likelihood...
datafiles = ["data/lightcone_mcmc_ns_data_%s.npz"%i for i in range(20)]
likelihood = p21mc.Likelihood1DPowerLightcone(  # All likelihood modules are prefixed by Likelihood*
    datafile = datafiles,        # All likelihoods have this, which specifies where to write/read data
    logk=False,                 # Should the power spectrum bins be log-spaced?
    min_k=0.02,                  # Minimum k to use for likelihood
    max_k=3.2,                  # Maximum ""
    nchunks = 11,                 # Number of chunks to break the lightcone into
    simulate=True
) # For other available options, see the docstring

model_name = "LightconeTest"

model_name = "LightconeTest_ns"
mcmc_options = {'n_live_points': 600,               # total number of live points
                'importance_nested_sampling': True, # do Importance Nested Sampling (INS)?
                'sampling_efficiency': 0.8,         # 0.8 and 0.3 are recommended for parameter estimation & evidence evalutaion respectively.
                'evidence_tolerance': 0.5,          # A value of 0.5 should give good enough accuracy.
                'max_iter': 20000,                  # maximum iteration
                'multimodal': True,                 # do mode separation?
                }

chainLF = mcmc.run_mcmc(
    core, # Assuming no model uncertainties.
    likelihood, # No need for datafile or noisefile if using the provided data.
    model_name=model_name,   # Filename of main chain output
    params=dict(             # Parameter dict as described above.
        HII_EFF_FACTOR = [30.0, 10.0, 200.0, 3.0],
        ION_Tvir_MIN = [4.7, 4, 6, 0.5],
        L_X = [40,38,42,1],
        NU_X_THRESH = [500,100,1200,100]
    ),
    datadir = 'data_ns',
    log_level_stream=logging.DEBUG,
    use_multinest=True,
    continue_sampling=False,
    **mcmc_options
)


