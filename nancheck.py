import py21cmfast as p21c
from py21cmfast import plotting
import h5py
import sys, os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

def main(argv):
    if len(argv) != 2:
        print("""Pass the path to the lightcones as an argument. (python nancheck.py "PATH_TO_FILES")""")
        return "ERROR"
    path = argv[1]
    iterlen = len(os.listdir(path))
    print(f"{iterlen}")
    for i,file in enumerate(os.listdir(path)):
        lc = p21c.outputs.LightCone.read(path + file)
        if np.isnan(lc.brightness_temp).any():
            #sleep(0.01)
            print(np.where(np.isnan(lc.brightness_temp)==True))
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
            plotting.lightcone_sliceplot(lc, ax=ax, fig=fig)
            ax.set_title(f"ION_Tvir_MIN={lc.astro_params.ION_Tvir_MIN}" + 
                        f"\nHII_EFF_FACTOR={lc.astro_params.HII_EFF_FACTOR}"+
                        f"\nL_X={lc.astro_params.L_X}"+
                        f"\nNU_X_THRESH={lc.astro_params.NU_X_THRESH}"+
                        f"\nrandom_seed={lc.random_seed}")
            fig.savefig(file+"nanplot.jpg")
            print(
                f"ION_Tvir_MIN={lc.astro_params.ION_Tvir_MIN}" + 
                        f"\nHII_EFF_FACTOR={lc.astro_params.HII_EFF_FACTOR}"+
                        f"\nL_X={lc.astro_params.L_X}"+
                        f"\nNU_X_THRESH={lc.astro_params.NU_X_THRESH}"+
                        f"\nrandom_seed={lc.random_seed}"
            )
        #print(f"{i}/{iterlen}", end='', flush=True)
        #print("\r", end='')

main(sys.argv)