import py21cmfast as p21c
from matplotlib import pyplot as plt
import os 
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.DEBUG)
from py21cmfast import plotting
from py21cmfast import cache_tools
from multiprocessing import Pool
#import timeit
import numpy as np
import random
import yaml
import itertools
from powerbox.tools import get_power
from schwimmbad import MPIPool
import h5py
import fnmatch
from collections.abc import Callable
from p21cmfastwrapper import Parameters
from numpy.typing import NDArray

class Leaf():
    def __init__(self, data_path: str = "./data/", data_prefix: str = "simrun_", parameter_file: str = None, 
                 cache_path: str = None, debug: bool = False) -> None:
        """
        Initializes the Leaf class with specified parameters.

        Args:
            data_path (str): The path where output will be saved. Default is "./data/"

            data_prefix (str): The prefix for the output files saved in the specified data_path. Default is "simrun_".
            
            parameter_file (str): The file from which to load parameters for all runs. This allows for more control over specific parameters.
                                  If None, standard parameters are used. Default is None.
            
            cache_path (str): Path for the 21cmFAST cache. If None, don't write cache. This can be faster if fast IO is available, but requires significant memory for large runs.
                                Recommended to use only if you plan to rerun simulations with the same parameters. Default is False.
            
            debug (bool): If True, enables verbose output to help identify errors. Default is False.
        """
        
        # define global variables
        self.data_path = data_path
        self.data_prefix = data_prefix
        if cache_path is None:
            self.write_cache = False
        else:
            self.write_cache = True
        self.debug = debug

        # make cache dir
        cache_path = "./_cache"
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        p21c.config['direc'] = cache_path
        # cache_tools.clear_cache(direc=cache_path) # <- clear cache

        if self.debug: print("Define global parameters...")

        self.astroparams = p21c.inputs.AstroParams
        self.cosmoparams = p21c.inputs.CosmoParams
        self.flagparams = p21c.inputs.FlagOptions
        self.globalparams = p21c.inputs.GlobalParams
        self.userparams = p21c.inputs.UserParams

        # init satistics
        self.nancounter = []
        self.tau


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

    def run_box(self, redshift: float, save: bool = True, random_seed: int = None, 
                sanity_check: bool = True, make_statistics: bool = True,
                astro_params: dict = None, cosmo_params: dict = None, user_params: dict = None,
                flag_options: dict = None, global_params: dict = None,
                run_id: int = 0) -> object | None:
        '''Run a coevel box of 21cmFAST given the parameters.
        
        Args:
            redshift (float): Redshift at which the box will be evaluated

            save (bool): If True, saves the result as a .h5, else returns the result

            random_seed (int): Pass a random seed to the simulator, if none it will be chosen randomly

            sanity_check (bool): Corrects for NaNs (NN-interpolation)

            make_statistics (bool); If True, saves interesting statistics about the box

            **params: Current parameters for the simulation
        '''
        self.debug("Begin box simulation ...")
        self.refresh_params(self, astro_params = astro_params, cosmo_params = cosmo_params,
                            user_params = user_params, flag_options = flag_options, global_params = global_params)
        self.debug("Parameter successfully refreshed.")    
        self.debug("Current parameters are:\n" + 
                   "astro_params: " + str(astro_params) + "\n" + 
                   "cosmo_params: " + str(cosmo_params) + "\n" + 
                   "user_params: " + str(user_params) + "\n" + 
                   "flag_options: " + str(flag_options) + "\n" + 
                   "global_params: " + str(global_params))
        with self.globalparams.use():
            run = p21c.run_coeval(redshift=redshift, astro_params = self.astroparams, 
                            cosmo_params = self.cosmoparams,
                            user_params = self.userparams, flag_options = self.flagparams, 
                            global_params = self.globalparams, random_seed=random_seed, write=self.write_cache)
            if sanity_check:
                run.brightness_temp = self.nan_adversarial(run.brightness_temp, run_id, debug)
            if make_statistics:
                # compute tau   
                self.tau.append(
                    p21c.compute_tau(redshifts=run.node_redshifts[::-1],
                                     global_xHI=run.global_xH[::-1]))
            if save:
                Leaf.save(run, fname=self.data_prefix, direc=self.data_path, run_id=run_id)
            else:
                return run           

    def refresh_params(self, astro_params: dict = None, cosmo_params: dict = None, user_params: dict = None,
                        flag_options: dict = None, global_params: dict = None) -> None:
        '''Update parameters'''
        self.astroparams.update(astro_params)
        self.cosmoparams.update(cosmo_params)
        self.userparams.update(user_params)
        self.flagparams.update(flag_options)
        self.globalparams.update(global_params)

    

    # utility function
    def debug(self, msg: str = "") -> None:
        if self.debug: print(msg)
    

    def nan_adversarial(self, bt_cone: NDArray, run_id: int, debug: bool) -> NDArray:
        nans = np.isnan(bt_cone)
        x_dim, y_dim, z_dim = bt_cone.shape
        if nans.any():
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
    
    @staticmethod
    def save(obj: object, fname: str, direc: str, run_id: int|str) -> None:
        obj.save(fname=fname+str(run_id), direc=direc)
    
    @staticmethod
    def load(path_to_obj: str, lightcone: bool) -> object:
        return p21c.outputs.LightCone.read(path_to_obj) if lightcone else p21c.outputs.Coeval.read(path_to_obj)