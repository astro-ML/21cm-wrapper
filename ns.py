
from mcmc import *
init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [10, 250],
                    "L_X": [38,42], "NU_X_THRESH": [100, 1500]}}

# input: init cube for parameter
def prior_sampling(theta):
        T_vir, H_eff, LX, nu_x = theta
        T_vir = 4 + 1.3*T_vir
        H_eff = 10 + 240*H_eff
        LX = 38 + 4*LX
        nu_x = 100 + 1400*nu_x
        return np.array([T_vir, H_eff, LX, nu_x])
    
# returns samples from the posterior
    
def log_likelihood(t,f):
    return - np.sum((t - f)**2/(np.abs(f)+1))


mcrun = mcmc(z_bins = 21, k_bins = 30, debug = False, log_likelihood=log_likelihood, prior=prior_sampling)
mcrun.make_fiducial(load = True)

mcrun.run_ns(init_params_ranges, name="./ns_chain1-")

