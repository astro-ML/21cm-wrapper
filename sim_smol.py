from Leaf import *
user_parameter = {
    "HII_DIM": 40,
    "BOX_LEN": 160,
    "N_THREADS": 1,
    "USE_INTERPOLATION_TABLES": False,
    "PERTURB_ON_HIGH_RES": True
}
flag_options = {
    "INHOMO_RECO": True,
    "USE_TS_FLUCT": True
}

astro_params = {
    "INHOMO_RECO": True
}

sim = Leaf(user_params=user_parameter, flag_options=flag_options, astro_params=astro_params, debug=True, data_path = "./data_smol/")

astro_params_range = {
        "HII_EFF_FACTOR":[10,250],
        "L_X":[38,42],
        "NU_X_THRESH":[100,1500],
        "ION_Tvir_MIN":[4.0,5.3]
}

cosmo_params_range = {"OMm":[0.2,0.4]}

global_params = {"M_WDM":[0.3,10.0]}  # M_WDM}



if __name__ == '__main__':
    sim.run_lcsampling(samplef=sim.uniform, redshift=5, save=True, threads=56, quantity=5000, filter_peculiar = True, sanity_check=True, make_statistics=True,
                    astro_params_range=astro_params_range, cosmo_params_range=cosmo_params_range, global_params_range=global_params)
