from p21cmfastwrapper import *
import emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import set_start_method
set_start_method("fork")

# self, parameter_path="./", save_inclass = False, save_ondisk = True, 
# write_cache=False, data_path = "./data/", file_name = "run_", override = False, 
# debug=False

class mcmc_aies(Simulation):
    def __init__(self, k_bins = np.linspace(2e-2, 3, 40), z_bins = 10, nwalker = 16, steps = 5000):
        super().__init__(save_inclass=False, save_ondisk = False, write_cache=False)
        self.k_bins = k_bins
        self.z_bins = z_bins
        self.nwalkers = nwalker
        self.nsteps = steps
        
    def make_fiducial(self, fparams, load=False):   
        if load: 
            self.fps = np.load("./fiducial_ps.npy")
        else:
            fcone = self.run_lightcone(kargs=fparams, commit=True)
            self.fps = self.compute_ps(fcone, self.z_bins, self.k_bins)
            np.save("./fiducial_ps.npy", self.fps)
        
    def init_params(self, init_params_ranges, samplef = (lambda a,b: np.uniform(a,b))):
        return self.generate_range(init_params_ranges, samplef)

    @staticmethod
    def compute_ps(data, z_bins, k_bins):
        zbins = np.linspace(0,data.lightcone_redshifts.shape[0]-1,z_bins).astype(int)
        ps = np.empty((z_bins.shape[0]-1, 3, k_bins.shape[0]-1))
        for bin in range(z_bins.shape[0] - 1):
            physical_size = data.lightcone_distances[zbins[bin+1]] - data.lightcone_distances[zbins[bin]]
            ps[bin,:,:] = get_power(deltax= data.brightness_temp[:,:,zbins[bin]:zbins[bin+1]], 
            boxlength=(*data.lightcone_dimensions[:2], physical_size), bin_ave=True, 
            ignore_zero_mode=True, get_variance=True, bins=k_bins, vol_normalised_power=True)
            ps[bin,0,:] *= ps[bin,1,:]**3/(2* np.pi**2)
        return ps

    # depricated, is supposed to be input for mcmc
    '''
    def log_prior(self, theta):
        xi, x_th, T_vir = theta
        if 5 < xi < 100 and 100 < x_th < 1500 and 4 < T_vir < 5.3:
            return 0
        return - np.inf
    '''
    
    def p_wrapper(self, theta, fid_ps, mc_parameter):
        # 21cmfast doesn't run outside of this ranges, implement generic hard-limit check in the future
        if (theta[1] < 100) | (theta[1] > 2000):
            return - np.inf 
        run_params = self.fill_dict(mc_parameter, theta)
        test_cone = self.run_lightcone(kargs=run_params, commit=True)
        test_ps = self.compute_ps(test_cone, self.z_bins, self.k_bins)
        

    @staticmethod
    def log_probability(theta, lp, llh):
        prob = self.log_prior(theta) + self.log_likelihood(theta, obs_box_ps, y)
        print("log_prob = ", prob)
        return prob

    @staticmethod
    def log_likelihood(x, y, cfunc):
        return cfunc(x,y)

    #multiprocessing.set_start_method("fork")
    def run(self):
        with Pool(int(self.nwalkers/2)) as pool:
            init = self.initialize()
            backend = emcee.backends.HDFBackend('./temp.h5')
            #pool.map = pool.imap_unordered
            y = [697.5588657421774, 631.7587919079817, 620.2692591040358, 611.818866161087, 615.0210167004124, 616.6042613052732, 635.1271091376823, 644.2463443515437, 650.9312828907001]
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool, args=([y]), backend=backend)
            sampler.run_mcmc(init, nsteps, progress=True)
        
    