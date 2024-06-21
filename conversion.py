import fnmatch, os 
import py21cmfast as p21c
import numpy as np

def convert_to_npz(path: str, prefix: str = "") -> None:
    '''Given a path and an optinal prefix 
    (e.g. only convert all files named as run_, set prefix = "run_")
    this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
    # image, label, tau, gxH
    # search for all files given in a path given a prefix an loop over those
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    len_files = len(files)
    # initalize the progress bar
    
    for i,file in enumerate(files):
        lcone = p21c.outputs.LightCone.read(file)
        # load image
        image = lcone.brightness_temp
        #load labels, WDM,OMm,LX,E0,Tvir,Zeta
        labels = [
            lcone.global_params["M_WDM"],
            lcone.cosmo_params.OMm,
            lcone.astro_params.L_X,
            lcone.astro_params.NU_X_THRESH,
            lcone.astro_params.ION_Tvir_MIN,
            lcone.astro_params.HII_EFF_FACTOR
        ]
        # load redshift
        redshifts = lcone.node_redshifts
        # compute tau
        gxH=lcone.global_xH
        gxH=gxH[::-1]
        redshifts=redshifts[::-1]
        tau=p21c.compute_tau(redshifts=redshifts,global_xHI=gxH)

        new_format = dict(
            "image": image,
            "label": labels,
            "tau": tau,
            "z": redshift,
            "gxH": gxH
        )
        #save to new format
        np.savez(file + ".npz", **new_format)
        
        # progress counter
        if i % (int(len_files/10)) == 0:
            idx = int(i / int(len_files) * 10)
            progress = "-"*(idx) + ">" + (10-idx)*" "
            print(f"|{progress}|", end='', flush=True)
            print("\r", end='')
        
        
        