from Flower import *
# initialize probability class

prior_ranges = {
    "global_params": {
        "M_WDM": [0.3,10.0]
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
        "N_THREADS": 6
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
    "cosmo_params":
        {"OMm": 0.30},
    "global_params":
        {"M_WDM": 2},
    "make_statistics": False,
}

if __name__ == '__main__':
    
    #bins = np.linspace(0.05, 3, 20)

    probability = Probability(prior_ranges=prior_ranges, z_chunks=20, bins=15, debug=False, 
                            fmodel_path="./emcee_data_1d/fiducial_ps.npy", summary_statistics='1dps')

    aies = Flower(Probability=probability, redshift=5, data_path="./emcee_data_1d/", noise_type=None,
                        regenerate_fiducial=False, fid_params=fiducial_parameter, debug=False)


    aies.run_emcee(threads=14, walkers = 56)
