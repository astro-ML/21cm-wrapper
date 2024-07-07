from Leaf import *
import emcee
import dynesty
from typing 

class KeyMismatchError(Exception):
    pass

class Probability:
    def __init__(self, prior_ranges: dict, chunks: int|list[int], bins: int, debug: bool = True, 
                 fmodel_path: str =  "./mcmc_data/fiducial_cone.npy"):
        """Stores the likelihood, priors and the summary statistics"""
        self.dodebug = debug
        self.chunks = chunks
        self.bins = bins
        self.fmodel_path = fmodel_path
        self.prior_ranges = prior_ranges

    def log_probability(self, parameters, lightcone):

        prob = - np.log(self.likelihood(lightcone=lightcone)) - np.log(self.prior(parameters=parameters))
        self.debug(f"Probability is {prob}")
        return prob
        

    def likelihood(self, lightcone: object):
        """Likelihood has lightcone object as input and outputs the likelihood"""
        fid_ps = np.load(self.fmodel_path)[:,0,:]
        test_ps = self.ps1d(lightcone=lightcone)[:,0,:]
        chi2 = self.loss(test_lc=test_ps, fiducial_lc=fid_ps)
        self.debug(f"Loss={chi2}")
        return chi2
        
    def prior(self, parameters: dict):
        dict1 = parameters
        dict2 = self.prior_ranges
        def compare_values(val1, range_tuple):
            return 1 if range_tuple[0] <= val1 <= range_tuple[1] else 0

        def compare_nested(dict1, dict2):
            result = {}
            for key, val in dict1.items():
                if key not in dict2:
                    raise KeyMismatchError(f"Key '{key}' found in dict1 but not in dict2")
                if isinstance(val, dict):
                    if not isinstance(dict2[key], dict):
                        raise KeyMismatchError(f"Value for key '{key}' is a dict in dict1 but not in dict2")
                    result[key] = compare_nested(val, dict2[key])
                else:
                    result[key] = compare_values(val, dict2[key])
            return result
            return 1
        
    def loss(self, test_lc, fiducial_lc):
        diff = test_lc - fiducial_lc
        return np.sum(np.divide(np.multiply(diff, diff), np.abs(fiducial_lc)+1))

    def ps1d(self, lightcone: object):
        """Compute 1D PS"""
        if type(self.chunks) == int:
            self.debug("Compute 1D PS using int chunks")
            zbins = np.linspace(0,lightcone.lightcone_redshifts.shape[0]-1,self.chunks+1).astype(int)
        else:
            self.debug("Compute 1D PS using list chunks")
            zbins = self.chunks
        ps = np.empty((len(zbins), 2, self.bins))
        for bin in range(len(zbins)):
            # get variance=False for now until nice usecase is found
            ps[bin,:,:] = get_power(deltax=lightcone.brightness_temp[:,:,zbins[bin]:zbins[bin+1]], 
            boxlength=lightcone.cell_size*np.array([*lightcone.brightness_temp.shape[:2], zbins[bin+1] - zbins[bin]]), 
            bin_ave=True, ignore_zero_mode=True, get_variance=False, bins=self.bins, vol_normalised_power=True)
            ps[bin,0,:] *= ps[bin,1,:]**3/(2* np.pi**2)
            self.debug(f"PS is {ps[bin,0,:]}" + f" for bin {bin}" + f"\nfor ks {ps[bin,1,:]}")
        return ps

    def debug(self, msg):
        if self.dodebug: print(msg)


class Simulation(Leaf):
    def __init__(self, Probability: Probability, redshift: float, data_path: str = "./mcmc_data/",
                 debug: bool = False, regenerate_fiducial: bool = True, **fid_params):
        """Stores everything related to the simulation like the fiducical model or noise"""
        # initialize lightcones with fid_params
        super().__init__(data_path=data_path, debug=debug, **fid_params)
        self.debug("initialize Simulation class...")
        self.redshift = redshift
        self.Probability = Probability

        if debug: print("Search for existing fiducial lightcone...")
        if (len(fnmatch.filter(os.listdir(self.data_path), "fiducial_cone.npy")) != 0) and not regenerate_fiducial:
            
            fiducial_cone = self.load(self.data_path + "fiducial_cone.npy", lightcone=True)
            if debug: print("Existing lightcone successfully loaded.")
        else:
            temp_threads = self.userparams.N_THREADS
            self.userparams.update(N_THREADS = os.cpu_count())
            fiducial_cone = self.run_lightcone(redshift=redshift, save=False, 
                                                random_seed=42, make_statistics=False)
            self.userparams.update(N_THREADS = temp_threads)
            self.save(obj=fiducial_cone, fname="fiducial_cone.npy", direc=data_path, run_id="")
            if debug: print("New lightcone successfully computed and saved.")
            
        if debug: print("Search for existing summary statistics...")
        if (len(fnmatch.filter(os.listdir(self.data_path), "fiducial_ps.npy")) != 0) and not regenerate_fiducial:
            self.fiducial_ps = self.load(self.data_path + "fiducial_ps.npy", lightcone=True)
            if debug: print("Existing summary statistics successfully loaded.")
        else:
            self.fiducial_ps = Probability.ps1d(fiducial_cone)
            self.save(obj=fiducial_cone, fname="fiducial_ps.npy", direc=data_path, run_id="")
            if debug: print("New summary statistics successfully computed and saved.")

    def step(self, parameters: list[float]):
        # convert parameter list to dict
        parameters = Simulation.replace_values(self.Probability.prior_ranges, parameters)
        self.debug("Current parameters are:", parameters)
        # run lightcone sim
        test_lc = self.run_lightcone(redshift=self.redshift, **parameters, make_statistics=False, 
                           sanity_check=False, filter_peculiar=True save=False)
        # compute probability
        return self.Probability.log_probability(lightcone=test_lc, parameters=parameters)
        
    @staticmethod   
    def replace_values(nested_dict, values_array):
        """
        Replace all values in the nested dictionary with the elements of the values_array in order.
        
        :param nested_dict: A dictionary which may contain other dictionaries as values.
        :param values_array: A list of values to replace in the nested dictionary.
        :return: A new dictionary with replaced values.
        :raises ValueError: If the number of values in the dictionary does not match the length of values_array.
        """
        def count_values(d):
            """ Recursively count the total number of values in the nested dictionary. """
            count = 0
            for value in d.values():
                if isinstance(value, dict):
                    count += count_values(value)
                else:
                    count += 1
            return count
        
        def replace(d, values):
            """ Recursively replace values in the nested dictionary with values from the list. """
            for key in d.keys():
                if isinstance(d[key], dict):
                    replace(d[key], values)
                else:
                    d[key] = values.pop(0)

        total_values = count_values(nested_dict)
        
        if total_values != len(values_array):
            raise ValueError("The number of values in the dictionary does not match the length of the values array.")
        replace(nested_dict, values_array)
        
        return nested_dict

class Flower(Simulation):
    def __init__(self, Probability: Probability, redshift: float, data_path: str = "./mcmc_data/",
                 debug: bool = False, regenerate_fiducial: bool = True, **fid_params):
        super().__init__(Probability = Probability, redshift = redshift, data_path = data_path,
                 debug = debug, regenerate_fiducial = regenerate_fiducial, **fid_params)
        self.Prob = Probability
        
    def run_emcee(self, )