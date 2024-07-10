import py21cmfast as p21c
from matplotlib import pyplot as plt
import os 
import logging, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
from py21cmfast import plotting
from py21cmfast import cache_tools
from concurrent.futures import ProcessPoolExecutor as Pool
from concurrent.futures import as_completed
from multiprocessing import get_context
#from multiprocessing import set_start_method, Pool
#import timeit
import numpy as np
import yaml
#from powerbox.tools import get_power
from schwimmbad import MPIPool
#import h5py
import fnmatch
from collections.abc import Callable
from typing import Generator
from p21cmfastwrapper import Parameters
from numpy.typing import NDArray
from alive_progress import alive_bar
import pickle


class Leaf():
    def __init__(self, data_path: str = "./data/", data_prefix: str = "simrun_", parameter_file: str = None, 
                 cache_path: str = None, debug: bool = False, redshift: float = None,
                astro_params: dict = {}, cosmo_params: dict = {}, user_params: dict = {},
                flag_options: dict = {}, global_params: dict = {}) -> None:
        """
        Initializes the Leaf class with specified parameters.

        Args:
            data_path: The path where output will be saved. Default is "./data/"

            data_prefix: The prefix for the output files saved in the specified data_path. Default is "simrun_".
            
            parameter_file: The file from which to load parameters for all runs. This allows for more control over specific parameters.
                                  If None, standard parameters are used. Default is None.
            
            cache_path: Path for the 21cmFAST cache. If None, don't write cache. This can be faster if fast IO is available, but requires significant memory for large runs.
                                Recommended to use only if you plan to rerun simulations with the same parameters. Default is False.
            
            debug: If True, enables verbose output to help identify errors. Default is False.

            redshift: The redshift at which the simulation ends

            **Parameter for 21cmFAST
        """
        
        # define global variables
        self.data_path = data_path
        self.data_prefix = data_prefix
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            
        if cache_path is None:
            self.write_cache = False
        else:
            self.write_cache = True
        self.dodebug = debug

        # make cache dir
        cache_path = "./_cache"
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        p21c.config['direc'] = cache_path
        # cache_tools.clear_cache(direc=cache_path) # <- clear cache

        if debug: print("Set initial parameters...")

        self.redshift = redshift
        self.astroparams = p21c.inputs.AstroParams(**astro_params)
        self.cosmoparams = p21c.inputs.CosmoParams(**cosmo_params)
        self.flagparams = p21c.inputs.FlagOptions(**flag_options)
        self.globalparams = global_params
        self.userparams = p21c.inputs.UserParams(**user_params)

        # init satistics
        self.nancounter = []
        self.tau = []
        self.filtercounter = []

        if parameter_file is not None:
            if self.debug: print("Use parameter file.")
            P = Parameters(parameter_path=parameter_file, file_name="./", data_path="./", override=False, debug=debug)
            input_params = P.give_all()
            self.astroparams(input_params['astro_params'])
            self.cosmoparams(input_params["cosmo_params"])
            self.userparams(input_params["user_params"])
            self.flagparams(input_params["flag_options"])
            self.globalparams(input_params["global_params"])
            if self.debug: print("Parameters from parameter file successfully loaded and set.")

    def run_box(self, redshift: float = None, save: bool = True, random_seed: int = None, 
                sanity_check: bool = True, make_statistics: bool = True,
                astro_params: dict = None, cosmo_params: dict = None, user_params: dict = None,
                flag_options: dict = None, global_params: dict = None,
                run_id: int = 0) -> object | None:
        '''Run a coevel box of 21cmFAST given the parameters.
        
        Args:
            redshift: Redshift at which the box will be evaluated

            save: If True, saves the result as a .h5, else returns the result

            random_seed: Pass a random seed to the simulator, if none it will be chosen randomly

            sanity_check: Corrects for NaNs (NN-interpolation)

            make_statistics: If True, saves interesting statistics about the box

            **params: Current parameters for the simulation
        '''
        self.debug("Begin box simulation ...")
        self.refresh_params(astro_params = astro_params, cosmo_params = cosmo_params,
                            user_params = user_params, flag_options = flag_options)
        if redshift is None: redshift = self.redshift
        self.debug("Parameter successfully refreshed.")    
        self.debug("Current parameters are:\n" + 
                   "astro_params: " + str(astro_params) + "\n" + 
                   "cosmo_params: " + str(cosmo_params) + "\n" + 
                   "user_params: " + str(user_params) + "\n" + 
                   "flag_options: " + str(flag_options) + "\n" + 
                   "global_params: " + str(global_params))
        with p21c.global_params.use(**global_params):
            run = p21c.run_coeval(redshift=redshift, astro_params = self.astroparams, 
                            cosmo_params = self.cosmoparams,
                            user_params = self.userparams, flag_options = self.flagparams, 
                            random_seed=random_seed, write=self.write_cache)
            if sanity_check:
                run.brightness_temp = self.nan_adversary(run.brightness_temp, run_id)
            if save:
                self.save(run, fname=self.data_prefix, direc=self.data_path, run_id=run_id)
            else:
                return run
            
    def run_lightcone(self, redshift: float = None, save: bool = True, random_seed: int = None, 
                sanity_check: bool = True, make_statistics: bool = False, filter_peculiar: bool = True,
                astro_params: dict = {}, cosmo_params: dict = {}, user_params: dict = {},
                flag_options: dict = {}, global_params: dict = {},
                run_id: int = 0) -> object | None:
        '''Run a coevel box of 21cmFAST given the parameters.
        
        Args:
            redshift: Redshift at which the box will be evaluated

            save: If True, saves the result as a .h5, else returns the result

            random_seed: Pass a random seed to the simulator, if none it will be chosen randomly

            sanity_check: Corrects for NaNs (NN-interpolation)

            make_statistics: If True, saves interesting statistics about the box

            filter_peculiar: see function lc_filter

            **params: Current parameters for the simulation
        '''
        self.debug("Begin lightcone simulation ...")
        self.refresh_params(astro_params = astro_params, cosmo_params = cosmo_params,
                            user_params = user_params, flag_options = flag_options)
        if redshift is None: redshift = self.redshift
        self.debug("Parameter successfully refreshed.")    
        self.debug("Current parameters are:\n" + 
                   "astro_params: " + str(self.astroparams) + "\n" + 
                   "cosmo_params: " + str(self.cosmoparams) + "\n" + 
                   "user_params: " + str(self.userparams) + "\n" + 
                   "flag_options: " + str(self.flagparams) + "\n" + 
                   "global_params: " + str(global_params))
        with p21c.global_params.use(**global_params):
            run = p21c.run_lightcone(redshift=redshift, astro_params = self.astroparams, 
                            cosmo_params = self.cosmoparams,
                            user_params = self.userparams, flag_options = self.flagparams, 
                            random_seed=random_seed, write=self.write_cache)
            self.debug("Done simulating. Do sanity check...")
            if sanity_check:
                run.brightness_temp = self.nan_adversary(run.brightness_temp, run_id)
            self.debug("Sanity check passed. Write statistics...")
            if make_statistics or filter_peculiar:
                tau = p21c.compute_tau(redshifts=run.node_redshifts[::-1],
                                     global_xHI=run.global_xH[::-1])
            if make_statistics:
                # compute tau   
                self.tau.append(tau)
            self.debug("Statistics written. Do filtering...")
            if filter_peculiar:    
                if not self.lc_filter(tau = self.tau[-1], gxH0=run.global_xH[-1], make_statistics=make_statistics, run_id=run_id):
                    return
            self.debug("Filtering passed. Save or return file now.")
            if save:
                self.save(obj=run, fname=self.data_prefix, direc=self.data_path, run_id=run_id)
            else:
                return run  
            

    def run_lcsampling(self, samplef: Callable, redshift: float = None, save: bool = True, random_seed: int = None, 
                sanity_check: bool = True, make_statistics: bool = True, filter_peculiar: bool = True,
                override: bool = False, threads: int = 1, mpi: bool = False, quantity: int = 1,
                astro_params_range: dict = {}, cosmo_params_range: dict = {}, user_params_range: dict = {},
                flag_options_range = {}, global_params_range: dict = {}) -> None:
        '''Run a coevel box of 21cmFAST given the parameters.
        
        Args:
            redshift (float): Redshift at which the box will be evaluated

            save (bool): If True, saves the result as a .h5, else returns the result

            random_seed (int): Pass a random seed to the simulator, if none it will be chosen randomly

            sanity_check (bool): Corrects for NaNs (NN-interpolation)

            make_statistics (bool): If True, saves interesting statistics about the box

            filter_peculiar (bool): see lc_filter

            override (bool): If True, override existing files

            threads (int): Define how many threads for multiprocessing will be used

            mpi (bool): If True, use mpi instead of Python's multiprocessing library (usually not worth it)

            quantity (int): Defines the amount of simulations being sampled

            *params_range: Give a dict consisting of the parameter as the key and a list passed to the samplefunction
                            e.g. astro_params = {HII_DIM: [140, 160]} for samplef = Leaf.uniform
        '''
        if astro_params_range == {} and cosmo_params_range == {} and user_params_range == {} and flag_options_range == {} and global_params_range == {}:
            print("No parameter ranges gives ... There is nothing to do.")
            return
        if redshift is None: redshift = self.redshift
        files = fnmatch.filter(os.listdir(self.data_path), self.data_prefix + "*")
        offset = 0 if override else len(files)
        run_ids = np.linspace(0,  quantity-1, quantity, dtype=int) + offset
        #for run_ids in Leaf.generate_run_ids(quantity=quantity, threads=quantity, offset=offset): <- is Generator hence cannot be pickled (I hate my life)
            # define the parameters
        runner = [{"redshift":redshift, "save":save, "random_seed":random_seed,
                                    "sanity_check":sanity_check, "make_statistics":make_statistics,
                                    "filter_peculiar":filter_peculiar, 
                                    "astro_params":self.generate_range(astro_params_range, samplef),
                                    "cosmo_params":self.generate_range(cosmo_params_range, samplef),
                                    "user_params":self.generate_range(user_params_range, samplef),
                                    "flag_options":self.generate_range(flag_options_range, samplef),
                                    "global_params":self.generate_range(global_params_range, samplef),
                                    "run_id":run_id} for run_id in run_ids]
        self.debug("Parameters:\n" + str(runner))
        # define pool type
        #set_start_method("spawn")
        schwimmhalle = MPIPool() if mpi else Pool(max_workers=threads, max_tasks_per_child=1, mp_context=get_context('spawn')) #Pool(maxtasksperchild=1, processes=threads)
        # run batch
        self.debug("Start running simulation...")
        with schwimmhalle as p:
            p.map(self.run_multilc, runner)

        if make_statistics:
            np.save(self.data_path + "statistics.npy", {
                "nancounter": self.nancounter,
                "tau": self.tau,
                "filtercounter": self.filtercounter
            })

            

    
    def run_multilc(self, kwargs): 
        '''Wrapper to pass dict of arguments to function with Python's multiprocessing lib'''
        return self.run_lightcone(**kwargs)


    def refresh_params(self, astro_params: dict = {}, cosmo_params: dict = {}, user_params: dict = {},
                        flag_options: dict = {}) -> None:
        '''Update parameters'''
        self.astroparams.update(**astro_params)
        self.cosmoparams.update(**cosmo_params)
        self.userparams.update(**user_params)
        self.flagparams.update(**flag_options)

    

    # utility function
    def debug(self, msg: str = ""):
        if self.dodebug: print(msg)
    

    def nan_adversary(self, bt_cone: NDArray, run_id: int) -> NDArray:
        nans = np.isnan(bt_cone)
        x_dim, y_dim, z_dim = bt_cone.shape
        if nans.any():
            self.debug("NaN(s) encountered at ", run_id, " count: ", len(np.where(nans == True)[0]))
            self.nancounter.append({
                "run_id": run_id,
                "astro_params": self.astroparams.defining_dict,
                "cosmo_params": self.cosmoparams.defining_dict,
                "flag_params": self.flagparams.defining_dict,
                "user_params": self.userparams.defining_dict
            })
            self.nancounter.append(run_id)
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
        
    def save(self, obj: object, fname: str, direc: str, run_id: int|str) -> None:
        self.debug(f"Save {run_id} to disk...")
        obj.save(fname=fname+str(run_id)+".h5", direc=direc)
    
    def load(self, path_to_obj: str, lightcone: bool) -> object:
        self.debug(f"Load {path_to_obj} from disk...")
        return p21c.outputs.LightCone.read(path_to_obj) if lightcone else p21c.outputs.Coeval.read(path_to_obj)

    def lc_filter(self, tau: float, gxH0: float, make_statistics: bool, run_id: int) -> bool:
        '''Apply tau and global nvalueeutral fraction at z=5 (gxH[0]) filters according to 
        https://github.com/astro-ML/3D-21cmPIE-Net/blob/main/simulations/runSimulations.py'''
        if tau>0.089 or gxH0>0.1: 
            self.debug("Lightcone rejected." + f" {tau=} " + f" {gxH0=} ")
            if make_statistics:
                self.filtercounter.append({
                    "run_id": run_id,
                    "astro_params": self.astroparams.defining_dict,
                    "cosmo_params": self.cosmoparams.defining_dict,
                    "flag_params": self.flagparams.defining_dict,
                    "user_params": self.userparams.defining_dict
                })
            return False
        else: return True

    ### Utility functions ### 

    def generate_range(self, nested_dict: dict, func: Callable) -> dict:
        '''Helper function which updates every values in a nested dict such that the values [a,b] -> func(*[a,b])'''
        if nested_dict is None:
            return None
        res = {} 
        for key, value in nested_dict.items(): 
            if isinstance(value, dict): 
                res[key] = self.generate_range(value, func) 
            else: 
                res[key] = func(*value)
        return res
    
    def fill_dict(self, nested_dict: dict, array: NDArray, index: int = 0) -> dict:
        '''Helper function to recursively fill a dict given an array'''
        for key in nested_dict:
            if isinstance(nested_dict[key], dict):
                index = self.fill_dict(nested_dict[key], array, index)
            else:
                if index < len(array):
                    nested_dict[key] = array[index]
                    index += 1
                else:
                    break
        return nested_dict
    
    def num_elements(self, x: dict) -> int:
        '''Helper function to recursively count the elements in a nested dict'''
        if isinstance(x, dict):
            return sum([self.num_elements(_x) for _x in x.values()])
        else: return 1
        
    def extract_values(self, nested_dict: dict) -> list[float]:
        '''Helper function to recursively extract all values from a nested dict'''
        values = []
        for key in nested_dict:
            if isinstance(nested_dict[key], dict):
                values.extend(self.extract_values(nested_dict[key]))
            else:
                values.append(nested_dict[key])
        return values
    
    def extract_keys(self, nested_dict: dict) -> list[str]:
        '''Helper function to recursively extract all keys from a nested dict'''
        keys = []
        for key in nested_dict:
            keys.append(key)
            if isinstance(nested_dict[key], dict):
                keys.extend(self.extract_keys(nested_dict[key]))
        return keys
    
    @staticmethod
    def generate_run_ids(quantity: int, threads: int, offset: int) -> Generator[NDArray, None, None]:
        counter = quantity
        while counter > 0:
                # make ranges and re-run multiprocessing for stability and performance
                counter -= threads
                if counter > 0:
                    yield np.linspace(quantity - (counter + threads),  quantity-counter-1, threads, dtype=int) + offset
                else:
                    yield np.linspace(quantity - (counter + threads),  quantity - 1, threads + counter, dtype=int) + offset

    # predefined samplefunctions
    @staticmethod
    def uniform(a,b): return np.random.uniform(a,b)
    @staticmethod
    def gauss(mu, sig): return np.random.gauss(mu, sig)
    @staticmethod
    def gumbel(loc, scale): return np.random.gumbel(loc, scale) # <- :3
    
    @staticmethod
    def plot_parameter_distribution(path: str = "./data/", prefix: str = "simrun_") -> None:

        files = fnmatch.filter(os.listdir(path), prefix + "*")


        OMm, HII_EFF_FACTOR, L_X, NU_X_THRESH, ION_Tvir_MIN  = ([] for _ in range(5))

        with alive_bar(len(files), force_tty=True) as bar:
            for file in files:
                lc = p21c.outputs.LightCone.read(fname=file, direc=path)
                OMm.append(lc.cosmo_params.OMm)
                HII_EFF_FACTOR.append(lc.astro_params.HII_EFF_FACTOR)
                L_X.append(lc.astro_params.L_X)
                NU_X_THRESH.append(lc.astro_params.NU_X_THRESH)
                ION_Tvir_MIN.append(lc.astro_params.ION_Tvir_MIN)
                bar()

        fig, ax = plt.subplots(3,2, figsize=(8,12))

        ax[0,0].hist(OMm, bins=20,density=True)
        ax[0,0].set_title("OMm")

        ax[0,1].hist(HII_EFF_FACTOR, bins=20,density=True)
        ax[0,1].set_title("HII_EFF_FACTOR")

        ax[1,0].hist(L_X, bins=20,density=True)
        ax[1,0].set_title("L_X")

        ax[1,1].hist(NU_X_THRESH, bins=20,density=True)
        ax[1,1].set_title("NU_X_THRESH")

        ax[2,0].hist(ION_Tvir_MIN, bins=20,density=True)
        ax[2,0].set_title("ION_Tvir_MIN")


        ax[2,1].axis("off")
        fig.tight_layout()
        fig.savefig("./data/parameter_distribution.png")
        fig.show()






# class from the older version to handle legacy loading via parameterfile
class Parameters():    
    '''Auxillary class to initialize and update parameters given a config file or on the fly.'''
    def __init__(self, parameter_path, file_name, data_path, override, debug):
        # set bool for box-lightcone-simulation switch
        self.box = False
        # load parameter file
        with open(parameter_path + "parameter.yaml", 'r') as file:
            parameter = yaml.safe_load(file)
        # set default configuration given by parameter file
        self.random_seed = not parameter["user_params"]["NO_RNG"]
        self.max_z = parameter["input_params"]["max_redshift"]
        use_default = []
        for key in parameter.keys():
            if key == "input_params":
                self.input_params = parameter["input_params"]
            else:
                print(f"""use {key} default config: {parameter[key]["use_default"]}""")
                use_default.append(parameter[key]["use_default"])
                parameter[key].pop("use_default")
        
        parameter.pop("input_params")
        
        self.astro_params = parameter["astro_params"] if not use_default[0] else p21c.AstroParams()
        self.cosmo_params = parameter["cosmo_params"] if not use_default[1] else p21c.CosmoParams()
        self.user_params = parameter["user_params"] if not use_default[2] else p21c.UserParams()
        self.flag_params = parameter["flag_options"] if not use_default[3] else p21c.FlagOptions()
        self.global_params = parameter["global_params"] if not use_default[4] else {}
        parameter.pop("global_params")
        
        self.input_params.update(parameter)
        
        # initialize saving procedure
        self.data_path = data_path
        self.data_name = file_name
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        # check if some files may already exist
        if override: os.system(f'rm -f {self.data_path + self.data_name + "*"}')
        if debug: print(self.data_path,self.data_name + "*",len(fnmatch.filter(os.listdir(self.data_path), self.data_name + "*")), fnmatch.filter(os.listdir(self.data_path), self.data_name + "*"))
        self.run_counter = 0 if override else len(fnmatch.filter(os.listdir(self.data_path), self.data_name + "*"))
        self.override = override
        
        # save the initial configuration
        self.init_params = self.input_params.copy() 

    def kwargs_update(self, kargs):
        '''Update the parameter config given kargs'''
        self.input_params.update(kargs)
        
    def kwargs_revert(self):
        '''Revert changes in the parameters'''
        self.input_params = self.init_params
    
    @staticmethod
    def wrap_params(params):
        '''Change the parameter file to run a single box and revert the changes afterwards.
        This is necessary or else 21cmfast returns an error.'''
        params.pop("lightcone_quantities")
        params.pop("max_redshift")
        return p21c.run_coeval(**params)
            
    def randomize(self):
        '''Shuffle random_seed'''
        if self.random_seed: self.input_params["random_seed"] = np.random.randint(0,99999)

    def give_all(self):
       return self.input_params