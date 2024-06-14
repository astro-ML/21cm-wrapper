from mcmc import *

init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [10, 250],
                    "L_X": [38,42], "NU_X_THRESH": [100, 1500]}}

def log_prior(theta):
        T_vir, H_eff, LX, nu_x = theta
        if 10 < H_eff < 250 and 100 < nu_x < 1500 and 4 < T_vir < 5.3 and 38 < LX < 42:
            return 0
        return - np.inf
    
def log_likelihood(t,f):
    return - np.sum((t - f)**2/(np.abs(f)+1))

mcrun = mcmc(nwalker=112, z_bins = 21, k_bins=30, 
             debug = False, log_likelihood=log_likelihood, prior=log_prior)
mcrun.make_fiducial(load = True)

mcrun.run_aies(init_params_ranges, mpi=False, threads=28, fname="./aies_chain1.h5")
