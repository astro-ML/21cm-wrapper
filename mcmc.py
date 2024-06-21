from p21cmfastwrapper import *
import emcee
from pymultinest.solve import solve
import json
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import set_start_method
set_start_method("spawn")
from schwimmbad import MPIPool

      

class mcmc(Simulation):
    def __init__(self, log_likelihood, prior, k_bins = np.linspace(5e-2,1.8, 30), z_bins = 11, nwalker = 16, steps = 5000, 
                 debug = False, nan_fix = True):
        super().__init__(save_inclass=False, save_ondisk = False, write_cache=False, clean_cache=False)
        self.k_bins = k_bins
        self.z_bins = z_bins
        self.nwalkers = nwalker
        self.nsteps = steps
        self.debug = debug
        self.llh = log_likelihood
        self.prior = prior
        self.nan_fix = nan_fix
        
    def make_fiducial(self, fparams: dict = {}, load: bool = False, plot=False):   
        if load: 
            self.fid_ps = np.load("./fiducial_ps.npy")
        else:
            fcone = self.run_cone(kargs=fparams, commit=True)
            self.fid_ps = self.compute_ps(fcone)
            np.save("./fiducial_ps.npy", self.fid_ps)
            if plot:
                print(self.fid_ps.shape)
                for bin in range(self.z_bins - 1):
                    plt.plot(self.fid_ps[bin,1,:], self.fid_ps[bin,0,:])
                    plt.show()
        
    def initialize_params(self, init_params_ranges: dict, samplef: Callable[[float, float], float] = (lambda a,b: np.random.uniform(a,b))):
        '''init_params_ranges: dict of ranges for initializing the walkers'''
        ini_params = np.empty((self.nwalkers, self.ndim))
        for i in range(self.nwalkers):
            ini_params[i] = self.extract_values(self.generate_range(init_params_ranges, samplef))        
        return ini_params

    def compute_ps(self, data):
        zbins = np.linspace(0,data.lightcone_redshifts.shape[0]-1,self.z_bins).astype(int)
        ps = np.empty((self.z_bins-1, 2, self.k_bins))
        for bin in range(self.z_bins - 1):
            physical_size = data.lightcone_distances[zbins[bin+1]] - data.lightcone_distances[zbins[bin]]
            # get variance=False for now until nice usecase is found
            ps[bin,:,:] = get_power(deltax= data.brightness_temp[:,:,zbins[bin]:zbins[bin+1]], 
            boxlength=(*data.lightcone_dimensions[:2], physical_size), bin_ave=True, 
            ignore_zero_mode=True, get_variance=False, bins=self.k_bins, vol_normalised_power=True)
            ps[bin,0,:] *= ps[bin,1,:]**3/(2* np.pi**2)
            if self.debug: print(f"{bin=}: ", f"{ps[bin,:,:]}")
        return ps
    
    def p_wrapper(self, theta, mc_parameter = None):
        if  self.debug: print()
        mc_parameter = self.mc_params if mc_parameter is None else mc_parameter
        # 21cmfast doesn't run outside of this ranges, implement generic hard-limit check in the future
        if (theta[3] < 100) | (theta[3] > 2000):
            if self.debug: print(f"{theta[3]=} out of range, return -inf")
            return - np.inf 
        run_params = self.fill_dict(mc_parameter, theta)
        if self.debug: print(f"{run_params=}")
        test_cone = self.run_cone(kargs=run_params, commit=True) 
        if self.nan_fix: test_cone.brightness_temp = self.nan_adversarial(test_cone.brightness_temp)
        #if np.isnan(test_cone.brightness_temp).any(): return - np.inf
        if self.debug: print(f"min/max b_temp: {test_cone.brightness_temp.min()}/", f"{test_cone.brightness_temp.max()}",f"\n{run_params=}")
        test_ps = self.compute_ps(test_cone)
        lprob = self.llh(test_ps[:,0,:], self.fid_ps[:,0,:]) if self.ns else self.log_probability(test_ps[:,0,:], self.fid_ps[:,0,:], theta)
        if self.debug: print(f"{lprob=}", f"\n{run_params=}")
        return lprob
        
    def log_probability(self, tps, fps, theta):
        p, l = self.prior(theta), self.llh(tps, fps)
        if self.debug: print(f"{p=}", "\n", f"{l=}")
        return p + l

    # lambda functions can not be pickled!, rewrite with normal functions, [x] done
    def run_aies(self, init_params_ranges,
            fname: str = "./aies_chain.h5", mpi = False, threads = 8):
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
        schwimmhalle = MPIPool() if mpi else Pool(threads)
        with schwimmhalle as pool:
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
    
    def sanity_check(self, params):
        not_sane = True
        while not_sane:
            data = self.run_cone(kargs=params, commit=True)
            not_sane = np.isnan(data.brightness_temp).any()
            print(f"{not_sane=}")
        return data

    @staticmethod
    def convert_to_npz(path: str, prefix: str = "", check_for_nan: bool = True, debug: bool = False) -> None:
        '''Given a path and an optinal prefix 
        (e.g. only convert all files named as run_, set prefix = "run_")
        this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
        # image, label, tau, gxH
        # search for all files given in a path given a prefix an loop over those
        files = fnmatch.filter(os.listdir(path), prefix + "*")
        if debug: print(f"{files}")
        len_files = len(files)
        nan_counter = []
        # initalize the progress bar
        
        for i,file in enumerate(files):
            lcone = p21c.outputs.LightCone.read(file)
            # load image
            image = lcone.brightness_temp
            # check if there are NaNs in the brightness map
            if check_for_nan:
                if np.isnan(image).any():
                    nan_counter.append(file)
                    continue
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

            new_format = {
                "image": image,
                "label": labels,
                "tau": tau,
                "z": redshifts,
                "gxH": gxH
            }
            #save to new format
            np.savez(file + ".npz", **new_format)
            
            # progress counter
            if len_files >= 10:
                if i % (int(len_files/10)) == 0:
                    idx = int(i / int(len_files) * 10)
                    progress = "-"*(idx) + ">" + (10-idx)*" "
                    print(f"|{progress}|", end='', flush=True)
                    print("\r", end='')
        print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")    

    @staticmethod
    def nan_adversarial(bt_cone):
        nans = np.isnan(bt_cone)
        x_dim, y_dim, z_dim = bt_cone.shape
        if nans.any():
            nan_idx = np.where(nans==True)
            for x,y,z in zip(*nan_idx):
                x_low, x_high = x-1, x+2
                y_low, y_high = y-1, y+2
                z_low, z_high = z-1, z+2
                if x == 0:
                    x_low += 1
                if x == x_dim -1:
                    x_high -= 1
                if y == 0:
                    y_low += 1
                if y == y_dim -1:
                    y_high -= 1
                if z == 0:
                    z_low += 1
                if z == z_dim -1:
                    z_high -= 1
                    
                region = bt_cone[x_low:x_high, y_low:y_high, z_low:z_high]
                bt_cone[x,y,z] = np.mean(region[~np.isnan(region)])
            return bt_cone       
        else:
            return bt_cone
            
