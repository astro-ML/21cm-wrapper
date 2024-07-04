from Leaf import *
user_parameter = {
    "HII_DIM": 30,
    "BOX_LEN": 150,
}
flag_options = {
    "INHOMO_RECO": False,
    "USE_TS_FLUCT": False
}

sim = Leaf(data_path = "./testdat/", user_params=user_parameter, flag_options=flag_options, debug=True)

astro_params_range = {
    "L_X": [39.42, 39.84],
    "HII_EFF_FACTOR": [29,29.5]

}





if __name__ == '__main__':
    sim.run_lcsampling(samplef=Leaf.uniform, redshift=5.5, save=True, threads=2, quantity=4,
                    astro_params_range=astro_params_range, sanity_check=True, filter_peculiar=False)
