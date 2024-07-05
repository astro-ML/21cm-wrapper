from Leaf import *
user_parameter = {
    "HII_DIM": 140,
    "BOX_LEN": 200,
    "N_THREADS": 2,
    "USE_INTERPOLATION_TABLES": True
}
flag_options = {
    "INHOMO_RECO": True,
    "USE_TS_FLUCT": True
}

astro_params = {
    "INHOMO_RECO": True
}

sim = Leaf(user_params=user_parameter, flag_options=flag_options, astro_params=astro_params, debug=True)

astro_params_range = {
        "HII_EFF_FACTOR":[10,250],
        "L_X":[38,42],
        "NU_X_THRESH":[100,1500],
        "ION_Tvir_MIN":[4.0,5.3]
}

cosmo_params_range = {"OMm":[0.2,0.4]}



if __name__ == '__main__':
    sim.run_lcsampling(samplef=sim.uniform, redshift=5.5, save=True, threads=28, quantity=2000,
                    astro_params_range=astro_params_range, cosmo_params_range=cosmo_params_range)