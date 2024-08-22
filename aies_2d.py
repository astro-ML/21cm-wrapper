# initialize probability class

from Flower import *

prior_ranges = {
    "global_params": {
        "M_WDM": [0.3,10]
    },
    "cosmo_params": {
        "OMm": [0.2,0.4]
    },
    "astro_params": {
        "L_X": [38, 42],
        "NU_X_THRESH": [100, 1500],
        "ION_Tvir_MIN": [4, 5.3],
        "HII_EFF_FACTOR": [10, 250]
    }
    
}

fiducial_parameter = {
    "user_params": {
        "HII_DIM": 40,
        "BOX_LEN": 160,
        "N_THREADS": 2,
        "USE_INTERPOLATION_TABLES": False,
    },
    "flag_options": {
        "USE_TS_FLUCT": True,
        "INHOMO_RECO": True
    },
    "astro_params": {
        "INHOMO_RECO": True,
        "L_X": 40,
        "NU_X_THRESH": 500,
        "ION_Tvir_MIN": 5,
        "HII_EFF_FACTOR": 30
    },
    "cosmo_params": {
        "OMm": 0.3
    },
    "global_params": {
        "M_WDM": 2.
    },
    "make_statistics": False,
}

if __name__ == "__main__":

    probability = Probability(prior_ranges=prior_ranges, z_eval=np.linspace(5.5,30, 25), bins=6, debug=True, 
                              fmodel_path="./data_emcee_2d/fiducial_ps.npy", summary_statistics='2dps')

    emcee = Flower(Probability=probability, redshift=5, data_path="./data_emcee_2d/", noise_type=None,
                           regenerate_fiducial=True, fid_params=fiducial_parameter, debug=True)

    emcee.make_fiducial()

    emcee.run_emcee(threads=3, nsteps=500, walkers=12)
