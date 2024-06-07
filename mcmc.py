from p21cmfastwrapper import *
import emcee
from pymultinest.solve import solve
import json
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import set_start_method
set_start_method("fork")

      

class mcmc(Simulation):
    def __init__(self, log_likelihood, prior, k_bins = np.linspace(2e-2, 3, 30), z_bins = 10, nwalker = 16, steps = 5000, 
                 debug = False):
        super().__init__(save_inclass=False, save_ondisk = False, write_cache=False)
        self.k_bins = k_bins
        self.z_bins = z_bins
        self.nwalkers = nwalker
        self.nsteps = steps
        self.debug = debug
        self.llh = log_likelihood
        self.prior = prior
        
    def make_fiducial(self, fparams: dict = {}, load: bool = False):   
        if load: 
            self.fid_ps = np.load("./fiducial_ps.npy")
        else:
            fcone = self.run_lightcone(kargs=fparams, commit=True)
            self.fid_ps = self.compute_ps(fcone)
            np.save("./fiducial_ps.npy", self.fid_ps)
        
    def initialize_params(self, init_params_ranges: dict, samplef: Callable[[float, float], float] = (lambda a,b: np.random.uniform(a,b))):
        '''init_params_ranges: dict of ranges for initializing the walkers'''
        ini_params = np.empty((self.nwalkers, self.ndim))
        for i in range(self.nwalkers):
            ini_params[i] = self.extract_values(self.generate_range(init_params_ranges, samplef))        
        return ini_params

    def compute_ps(self, data):
        zbins = np.linspace(0,data.lightcone_redshifts.shape[0]-1,self.z_bins).astype(int)
        ps = np.empty((self.z_bins-1, 2, self.k_bins.shape[0]-1))
        for bin in range(self.z_bins - 1):
            physical_size = data.lightcone_distances[zbins[bin+1]] - data.lightcone_distances[zbins[bin]]
            # get variance=False for now until nice usecase is found
            ps[bin,:,:] = get_power(deltax= data.brightness_temp[:,:,zbins[bin]:zbins[bin+1]], 
            boxlength=(*data.lightcone_dimensions[:2], physical_size), bin_ave=True, 
            ignore_zero_mode=True, get_variance=False, bins=self.k_bins, vol_normalised_power=True)
            ps[bin,0,:] *= ps[bin,1,:]**3/(2* np.pi**2)
        return ps
    
    def p_wrapper(self, theta, mc_parameter = None):
        mc_parameter = self.mc_params if mc_parameter is None else mc_parameter
        # 21cmfast doesn't run outside of this ranges, implement generic hard-limit check in the future
        if (theta[3] < 100) | (theta[3] > 2000):
            return - np.inf 
        run_params = self.fill_dict(mc_parameter, theta)
        test_cone = self.run_lightcone(kargs=run_params, commit=True)
        test_ps = self.compute_ps(test_cone)
        lprob = self.llh(test_ps[:,0,:], self.fid_ps[:,0,:]) if self.ns else self.log_probability(test_ps[:,0,:], self.fid_ps[:,0,:], theta)
        if self.debug: print(f"{lprob=}")
        return lprob
        
    def log_probability(self, tps, fps, theta):
        return self.prior(theta) + self.llh(tps, fps)

    # lambda functions can not be pickled!, rewrite with normal functions, [x] done
    def run_aies(self, init_params_ranges,
            fname: str = "./aies_chain.h5"):
        '''init_params_ranges: Sets the parameter ranges for initializing the walkers'''
        self.ns = False
        if init_params_ranges == None:
            # continue saved run
            print("Not implemented :(")
            raise RuntimeError
        self.ndim = self.num_elements(init_params_ranges)
        init = self.initialize_params(init_params_ranges)
        if self.debug: print(init)
        backend = emcee.backends.HDFBackend(fname)
        with Pool(int(self.nwalkers/2)) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.p_wrapper, pool=pool, args=([init_params_ranges]), backend=backend)
            sampler.run_mcmc(init, self.nsteps, progress=True)
            
    def run_ns(self, mc_params, name: str = "./ns_chain-"):
        self.ns = True
        self.mc_params = mc_params
        parameters = self.extract_keys(self.mc_params)
        ndim = self.num_elements(mc_params)
        result = solve(LogLikelihood=self.p_wrapper, Prior=self.prior,
                    n_dims=ndim, outputfiles_basename=name, verbose=True)
        if self.debug:
            print()
            print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
            print()
            print('parameter values:')
            for name, col in zip(parameters, result['samples'].transpose()):
                print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            with open('%sparams.json' % name, 'w') as f:
                json.dump(parameters, f, indent=2)