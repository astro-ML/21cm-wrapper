import py21cmfast as p21c
from matplotlib import pyplot as plt
import os 
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
from py21cmfast import plotting
from py21cmfast import cache_tools
#from multiprocessing import Pool
import timeit
import numpy as np
import random
import yaml
import itertools
from powerbox.tools import get_power
import pickle

# set your cache path here
cache_path = "../_cache"

if not os.path.exists(cache_path):
    os.mkdir(cache_path)

p21c.config['direc'] = cache_path
# uncomment this if you want to clear the cache everytime you execute the program
# cache_tools.clear_cache(direc=cache_path)

class Parameters():    
    '''Auxillary class to initialize and update parameters given a config file or on the fly.'''
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
            #self.z = [self.input_params["redshift"] + (self.input_params["max_redshift"] - self.input_params["redshift"])/zsteps * i for i in range(zsteps+1)] 
            self.z = np.linspace(self.input_params["redshift"], self.input_params["redshift"], zsteps).tolist()
            self.input_params["redshift"] = self.z
            self.input_params.pop("max_redshift")
        else:
            self.prechange = True
            self.input_params["lightcone_quantities"] = self.lcone_quantities
            self.input_params["redshift"] = self.z[0]
            self.input_params["max_redshift"] = self.z[-1]
            
    def randomize(self):
        self.input_params["random_seed"] = random.randint(0,99999)
                
            
        
    
class Simulation(Parameters):
    '''Dynamically execute and plot simulations.'''
    def __init__(self, random_seed=True):
        '''random_seed (bool): Isf true, each simulation is executed with a random generated seed. If you want to reproduce a simulation, set this to False.'''
        super().__init__(random_seed=random_seed)
        print(f"Using 21cmFAST version {p21c.__version__}")
        self.data = []
        self.randseed = random_seed
        
    def __len__(self):
        '''Returns length of the data-array'''
        return len(self.data)
        
    def run_box(self, runs=1, zsteps=1, kargs={}, nosave=False):
        '''Run a simple box simulation'''
        self.simtype = 0
        self.pbox_run(zsteps)
        with p21c.global_params.use(**self.global_params):
            for _ in range(runs):
                if self.randseed: self.randomize()
                self.kwargs_update(kargs)
                run = p21c.run_coeval(**self.input_params)
                if not nosave: 
                    self.data.append(run)
                    self.pbox_run(zsteps)
                else:
                    self.pbox_run(zsteps) 
                    return run
                
        
    
    def run_lightcone(self, runs=1, kargs={}):
        '''Run a simple lightcone simulation'''
        self.simtype = 1
        with p21c.global_params.use(**self.global_params):
            for _ in range(runs):
                if self.randseed: self.randomize()
                self.kwargs_update(kargs)
                self.data.append(p21c.run_lightcone(**self.input_params))
                
    def run_fixed_multi_lightcone(self, rargs):
        '''Compute multiple lightcones given a list of parameters as a dict with list entries
        e.g. rargs = {"random_seed": [1,2], "astro_params": {"HII_EFF_FACTOR": [29,31]}, NU_X_THRESH": [1,2,3]}, ...}'''
        for run_params in self.generate_combinations(rargs):
            print("Parameter run: ",run_params)
            self.run_lightcone(runs=1, kargs=run_params)
            
    def plot_global_properties(self, run=[-1], observational_axis = False, print_params = ['']):
        '''Make a plot of the global quantities of the lightcone
        run: array of run_ids which should be printed
        observational_axis: plot the x-axis in MHz instead of redshift
        print_params: Additional array of parameters which are printed as plain text for debug reasons'''
        x_map = (lambda z: 1420.4/(z + 1)) if observational_axis else (lambda z: z)
        if run[0] == -1:
            run = range(len(self.data))
        if self.simtype != 1:
            print("Non-lightcone simulation not supported.")
            return
        w,h = len(run), len(self.data[0].global_quantities)
        fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
        ax = ax.reshape(h,w)
        for r in run:
            add_string = ''
            for p in print_params:
                if p != '': add_string += p + '=' + str(self.data[r].astro_params.defining_dict[p]) + ', '
            title = f"run={r}"
            if add_string != '': title += '\n' + add_string
            ax[0,r].set_title(f"{title}")
            for q, quantity in enumerate(self.data[r].global_quantities.keys()):
                ax[q,r].plot(x_map(np.array(self.data[r].node_redshifts)), self.data[r].global_quantities[quantity])
                ax[q,r].set_xlabel("MHz" if observational_axis else "z")
                ax[q,r].set_ylabel(quantity)
        plt.tight_layout()
        plt.savefig("./glob_prop_test.jpg")
        plt.show()
            
    def plot_imshow(self, run=-1, fields=["brightness_temp"], print_params = [''], extended=True):
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
                for field in fields:
                    fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
                    ax = ax.reshape(h,w)
                    for (j, axis) in zip(range(len(data)),  ax):
                        add_string = ''
                        for p in print_params:
                            if p != '': add_string += p + '=' + str(data[j].astro_params.defining_dict[p]) + ', '
                        for i, (dat, redshift, a) in enumerate(zip(data[j], [z.redshift for z in self.data[j]], axis)):
                            if i<1:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field)
                            else:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field, printlabel=False)
                            title = f"{field}, z = {round(redshift,2)}"
                            if add_string != '': title += '\n' + add_string
                            plt.title(title)
                    plt.tight_layout()
                    plt.savefig("./imshow_test.jpg")
                    plt.show()
            case 1:
                h = len(data)
                for field in fields:
                    fig, ax = plt.subplots(h,1, figsize=(12,4*h))
                    ax = ax.flatten() if h > 1 else [ax]
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
                    plt.savefig("./imshow_test.jpg")
                    plt.show()     
                    
    def plot_ps(self, bins=10, run=[-1], compare=False, observational_axis = False, logp = True, z_bins=50):
        '''Plot the power spectra given a fixed bin number. (Future: Do the same with a dz range but p21cm saves lightcone with constant physical distances not redshift ->)
        run (array): plot ps of given run, -1 to print them all
        compare (bool): Activating compare puts all simulations with equal redshift range in a single plot (useful to compare the ps of different simualtions)
        in_f (bool): changes the x axis from k in Mpc^-3 to lambda in real space (Mpc^3)
        logp (bool): Activating make the plots in log-log axis.
        '''
        binspace = np.linspace(0,len(self.data[0].lightcone_redshifts)-1,bins).astype(int)
        
        x_map = (lambda k: 2*np.pi/k) if observational_axis else (lambda k: k)
        
        if run[0] == -1:
            run = range(len(self.data))
            
            
        if not compare: 
            h,w = len(run),bins-1
            fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
            ax = ax.reshape(h,w)
            for r in run:
                for bin in range(w):
                    physical_size = self.data[r].lightcone_distances[binspace[bin+1]] - self.data[r].lightcone_distances[binspace[bin]]
                    #print(f"binspace: {binspace[bin]}, brightness map: {self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]].shape}, boxlength= {(*self.data[r].brightness_temp.shape[:2], physical_size)}")
                    ps = get_power(deltax= self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]], boxlength=(*self.data[r].lightcone_dimensions[:2], physical_size), bin_ave=True, ignore_zero_mode=True, get_variance=False, bins=z_bins)
                    ax[r,bin].plot(x_map(ps[1]), ps[0]*ps[1]**3)
                    ax[-1,bin].set_xlabel('m in Mpc^3' if observational_axis else "k in Mpc^-3")
                    ax[0,bin].set_title(f'{round(self.data[r].lightcone_redshifts[binspace[bin]],1)} < z < {round(self.data[r].lightcone_redshifts[binspace[bin+1]],1)}')
                    ax[r,0].set_ylabel(f"P(k) * k ^ 3 ; run: {r}")
                    if logp:
                        ax[r,bin].set_xscale("log")
                        ax[r,bin].set_yscale("log")
        else:
            w = bins - 1 
            fig, ax = plt.subplots(run,figsize=(4*run,4))
            for r in run:
                for bin in range(w):
                    physical_size = self.data[r].lightcone_distances[binspace[bin+1]] - self.data[r].lightcone_distances[binspace[bin]]
                    ps = get_power(deltax= self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]], boxlength=(*self.data[r].lightcone_dimensions[:2], physical_size), bin_ave=True, ignore_zero_mode=True, get_variance=False,bins=z_bins)
                    print(ps[1])
                    ax.plot(x_map(ps[1]), ps[0]*ps[1]**3, label=f"{round(self.data[r].lightcone_redshifts[binspace[bin]],1)} < z < {round(self.data[r].lightcone_redshifts[binspace[bin+1]],1)}")
                    #ax.set_title(f'{round(self.data[r].lightcone_redshifts[binspace[bin]],1)} < z < {round(self.data[r].lightcone_redshifts[binspace[bin+1]],1)}')
                    ax.set_xlabel('m in Mpc^3' if observational_axis else "k in Mpc^-3")
                    ax.set_ylabel(f"P(k) * k ^ 3 ; run: {r}")
                    ax.legend()
                    if logp:
                        ax.set_xscale("log")
                        ax.set_yscale("log")
        
        plt.tight_layout()
        plt.savefig("./ps_test.jpg")
        plt.show()
        
    # Helper function to recursively generate combinations
    def generate_combinations(self, d):
        keys, values = zip(*d.items())
        # For each value, if it's a dict, recursively call generate_combinations
        values = [self.generate_combinations(v) if isinstance(v, dict) else v for v in values]
        # Generate all combinations using itertools.product
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
        return self.generate_combinations(d)

    def pop(self,idx):
        '''Delete a run with idx'''
        self.data.pop(idx)
        
    def clear(self):
        '''Clear the data/runs cache'''
        self.data.clear()
        
    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.data, f)
        
    def load(self, name):
        with open(name, 'rb') as f:
            self.data = pickle.load(f)