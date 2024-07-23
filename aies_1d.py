from Flower import *

# initialize probability class

prior_ranges = {
    "astro_params": {
        "L_X": [38, 42],
        "NU_X_THRESH": [100, 1500],
        "ION_Tvir_MIN": [4, 5.3],
        "HII_EFF_FACTOR": [10, 250]
    }
    "cosmo_params": {
        "OMm": 
    }
}

fiducial_parameter = {
    "user_params": {
        "HII_DIM": 40,
        "BOX_LEN": 160,
        "N_THREADS": 1,
        "USE_INTERPOLATION_TABLES": True,
    },
    "flag_options": {
        "USE_TS_FLUCT": False,
        "INHOMO_RECO": False
    },
    "astro_params": {
        "INHOMO_RECO": False,
        "L_X": 40,
        "NU_X_THRESH": 500,
        "ION_Tvir_MIN": 5,
        "HII_EFF_FACTOR": 30
    },
    "make_statistics": False,
}

probability = Probability(prior_ranges=prior_ranges, z_chunks=10, bins=10, debug=True, 
                          fmodel_path="./emcee_data/fiducial_ps.npy")

emcee = Flower(Probability=probability, redshift=5, data_path="./emcee_data/", noise_type=None,
                       regenerate_fiducial=True, fid_params=fiducial_parameter, debug=False)


emcee.run_emcee(threads=12, nsteps=240, walkers=24)
