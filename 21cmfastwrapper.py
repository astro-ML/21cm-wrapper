import py21cmfast as p21c
from matplotlib import pyplot as plt
import os 
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
from py21cmfast import plotting
from py21cmfast import cache_tools
from multiprocessing import Pool
import timeit
import numpy as np
import random
import yaml
from powerbox.tools import get_power

cache_path = "../_cache"

if not os.path.exists(cache_path):
    os.mkdir(cache_path)

p21c.config['direc'] = cache_path
# cache_tools.clear_cache(direc=cache_path)

print(f"Using 21cmFAST version {p21c.__version__}")

class parameters():    
    '''Initialize parameter given a config file'''
    def __init__(self, random_seed=True, parameter_path="./"):
        self.prechange = True
        self.random_seed = random_seed
        with open(parameter_path + "parameter.yaml", 'r') as file:
            parameter = yaml.safe_load(file)
        
        
        
        use_default = []
        for key in parameter.keys():
            if key == "input_params":
                self.input_params = parameter["input_params"]
            else:
                print(f"use {key} default config: {parameter[key]["use_default"]}")
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
        self.standard_config = self.input_params.copy() 
        
    def kwargs_update(self, kargs):
        self.input_params.update(kargs)
        
            
    def pbox_run(self, zsteps):
        if self.prechange:
            self.prechange = False
            self.lcone_quantities = self.input_params.pop("lightcone_quantities")
            self.z = [self.input_params["redshift"] + (self.input_params["max_redshift"] - self.input_params["redshift"])/zsteps * i for i in range(zsteps+1)] 
            self.input_params["redshift"] = self.z
            self.input_params.pop("max_redshift")
        else:
            self.prechange = False
            self.input_params["lightcone_quantities"] = self.lcone_quantities
            self.input_params["redshift"] = self.z[0]
            self.input_params["max_redshift"] = self.z[-1]
            
    def randomize(self):
        self.input_params["random_seed"] = random.randint(0,99999)
                
            
    
        
        
    
class simulation(parameters):
    '''Dynamically execute and plot simulation'''
    def __init__(self, random_seed=True):
        super().__init__(random_seed=random_seed)
        self.data = []
        
    def __len__(self):
        return len(self.data)
        
    def run_box(self, runs, zsteps, kargs={}):
        self.simtype = 0
        self.pbox_run(zsteps)
        with p21c.global_params.use(**self.global_params):
            for _ in range(runs):
                self.kwargs_update(kargs)
                self.data.append(p21c.run_coeval(**self.input_params))
        self.pbox_run(zsteps)
    
    def run_lightcone(self, runs, kargs={}):
        self.simtype = 1
        with p21c.global_params.use(**self.global_params):
            for _ in range(runs):
                self.kwargs_update(kargs)
                self.data.append(p21c.run_lightcone(**self.input_params))
        
    def plot(self, run=-1, fields=["brightness_temp"], print_params = [''], extended=True):
        '''
        run: plot given run, -1 to print them all
        fiels: list of fields that should be printed
        print_params: also show given parameters on the plots (only astro parameter!)
        extended: make additional graphs and statistics'''
        if run == -1:
            data = self.data
        else: 
            data = [self.data[run]]
        match self.simtype:
            case 0:
                h = len(data)
                w = len(data[0])
                fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
                ax = ax.reshape(h,w)
                for field in fields:
                    for (j, axis) in zip(range(len(data)),  ax):
                        add_string = ''
                        for p in print_params:
                            if p != '': add_string += p + '=' + str(data[j].astro_params.defining_dict[p]) + ', '
                        for i, (dat, redshift, a) in enumerate(zip(data[j], [z.redshift for z in test.data[j]], axis)):
                            if i<1:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field)
                            else:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field, printlabel=False)
                            title = f"{field}, z = {round(redshift,2)}"
                            if add_string != '': title += '\n' + add_string
                            plt.title(title)
                    plt.tight_layout()
                    plt.show()
            case 1:
                h = len(data)
                fig, ax = plt.subplots(h,1, figsize=(12,4*h))
                ax = ax.flatten() if h > 1 else [ax]
                for field in fields:
                    for i, dat in enumerate(data):
                        add_string = ''
                        for p in print_params:
                            if p != '': add_string += p + " = " + str(data[i].astro_params.defining_dict[p]) + ', '
                        z = dat.node_redshifts
                        plotting.lightcone_sliceplot(dat, ax=ax[i], fig=fig, kind=field)
                        title = f"{field}, z = [{round(max(z),2)} -> {round(min(z),2)}] in {len(z)} steps, run = {i+1} "
                        if add_string != '': title += '\n' + add_string
                        plt.title(title)
                    ax = np.reshape(ax, (h,1))
                    plt.tight_layout()
                    plt.show()     
        

        
    def pop(self,idx):
        self.data.pop(idx)
        
    def clear(self):
        self.data.clear()