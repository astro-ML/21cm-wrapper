
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
        HII_DIM = 70,
        BOX_LEN = 200.0,
        PERTURB_ON_HIGH_RES = True,
        USE_INTERPOLATION_TABLES = False,
	N_THREADS=2),
    flag_options = dict(
        INHOMO_RECO = True,
        USE_TS_FLUCT = True,
    ),
    regenerate=False,
    direc="_cache",
    write=False,) # For other available options, see the docstring.

# Now the likelihood...
datafiles = ["data_emcee/lightcone_mcmc_data_%s.npz"%i for i in range(11)]
likelihood = p21mc.Likelihood1DPowerLightcone(  # All likelihood modules are prefixed by Likelihood*
    datafile = datafiles,        # All likelihoods have this, which specifies where to write/read data
    logk=False,                 # Should the power spectrum bins be log-spaced?
    min_k=0,                  # Minimum k to use for likelihood
    max_k=3.4,                  # Maximum ""
    nchunks = 11,                 # Number of chunks to break the lightcone into
    simulate=True,
) # For other available options, see the docstring

model_name = "LightconeTest"

chain = mcmc.run_mcmc(
    core, likelihood,        # Use lists if multiple cores/likelihoods required. These will be eval'd in order.
    datadir='data_emcee',          # Directory for all outputs
    model_name=model_name,   # Filename of main chain output
    params=dict(             # Parameter dict as described above.
        HII_EFF_FACTOR = [30.0, 10.0, 200.0, 3.0],
        ION_Tvir_MIN = [4.7, 4, 6, 0.5],
        L_X = [40,38,42,1],
        NU_X_THRESH = [500,100,1200,100]
    ),
    walkersRatio=13.5,         # The number of walkers will be walkersRatio*nparams
    burninIterations=0,      # Number of iterations to save as burnin. Recommended to leave as zero.
    sampleIterations=100,    # Number of iterations to sample, per walker.
    threadCount=27,           # Number of processes to use in MCMC (best as a factor of walkersRatio)
    log_level_stream=logging.DEBUG,
    log_level_21CMMC=logging.DEBUG,
    continue_sampling=False,  # Whether to contine sampling from previous run *up to* sampleIterations.
)

