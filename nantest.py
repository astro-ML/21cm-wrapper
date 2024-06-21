import py21cmfast as p21c
p21c.cache_tools.clear_cache(direc="_cache")

user_params = p21c.UserParams(
    HII_DIM=120, BOX_LEN=200, NO_RNG=True, USE_INTERPOLATION_TABLES=False, PERTURB_ON_HIGH_RES=True, N_THREADS= 6
)
flag_options = p21c.FlagOptions(
    INHOMO_RECO=True, USE_TS_FLUCT=True
)

astro_params = p21c.AstroParams(
    **{'ION_Tvir_MIN': 4.798687324224536, 'HII_EFF_FACTOR': 153.93554078107047, 'L_X': 41.07914816577325, 'NU_X_THRESH': 122.31064697905447}
)

lightcone = p21c.run_lightcone(
    direc='_cache',
    user_params=user_params,
    flag_options=flag_options,
    astro_params=astro_params,
    redshift = 5.5
)

print(np.where(np.isnan(lightcone.brightness_temp)==True))
