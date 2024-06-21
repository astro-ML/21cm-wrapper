from p21cmfastwrapper import *

sim = Simulation(save_ondisk=True, save_inclass=False, write_cache=False, clean_cache=False, data_path = "/remote/gpu01a/schlenker/21cm-wrapper/data2/")

args = {"M_WDM":[0.3,10.0],
                "astro_params": {"HII_EFF_FACTOR":[10,250],"L_X":[38,42],"NU_X_THRESH":[100,1500],"ION_Tvir_MIN":[4.0,5.3]},
                "cosmo_params": {"OMm":[0.2,0.4]}}
sim.run_samplef(nruns=50000, args=args, threads = 28)
