Leaf wrapper for 21cmFAST and more v1.0 <br/>
is supposed to provide an easy, high-level interface written to <br/>
minimize pain when doing doing simulations or other things with 21cmFAST.<br/>
It also provides the user with a fast multiprocessing interface for large-scale <br/>
databases with the option to also control low-level parameters.  
![alt text](https://github.com/astro-ML/21cm-wrapper/blob/main/Leaf.png?raw=true)


<b>Quickstart</b> <br/>
(* - required for cluster deployment):<br/> <br/>
Run ```install.sh``` or alternatively follow the step-by-step guide below. <br/> <br/>
1*) Create an new virtual environment <br />
```python -m venv ./21cm-wrapper-penv```<br /><br />
2*) Activate the environment <br />
```source ./21cm-wrapper-penv/bin/activate``` <br /><br />
3) Install the requirements <br />
```pip install -r requirements.txt``` <br /><br />
4) Get familiar with the wrapper by open tutorial.ipynb in your jupyter notebook <br />
  You can import the classes and load all dependencies by running <br />
```from Leaf import *``` <br /><br />
5) Give initialization parameter which are fixed for all simulations or edit the parameter.yaml file to your liking.<br /> <br /> <br />

<b>Data generation on cluster</b> <br/>
For running the code on a cluster using PBS one needs to create two files:<br /> <br />

1) A python script which will be executed. For running simulations with uniform sampled parameters<br />
  the script may look like this: <br />
```python
#import the wrapper
from Leaf import *

# To set the parameter ranges, the parameter must be given as a dict, following the dict
# structure defined in parameter.yaml. The values must be an array which is handed over
# to the samplef defined above. In this case it is [start, stop].
user_parameter = {
    "HII_DIM": 140,
    "BOX_LEN": 200,
    "N_THREADS": 1,
    "USE_INTERPOLATION_TABLES": True,
    "PERTURB_ON_HIGH_RES": True
}
flag_options = {
    "INHOMO_RECO": True,
    "USE_TS_FLUCT": True
}

astro_params = {
    "INHOMO_RECO": True
}

redshift = 5.5

# Initialize the wrapper with the above given parameters,
# for more details check the docs or the tutorial.ipynb.
sim = Leaf(user_params=user_parameter, flag_options=flag_options, astro_params=astro_params, debug=True, redshift=redshift)

astro_params_range = {
        "HII_EFF_FACTOR":[10,250],
        "L_X":[38,42],
        "NU_X_THRESH":[100,1500],
        "ION_Tvir_MIN":[4.0,5.3]
}

cosmo_params_range = {"OMm":[0.2,0.4]}


# Set the number of simulations with quantity and the number of threads (= #cores on the cluster)
if __name__ == '__main__':
    sim.run_lcsampling(samplef=sim.uniform, save=True, threads=28, quantity=2000,
                    astro_params_range=astro_params_range, cosmo_params_range=cosmo_params_range)
# That's it!
```
And we save it as simulator.py. <br />  <br />

2) A bash script which is queued using the  <br />
```qsub``` <br />
command. It may look like this: <br />
```bash
#!/bin/bash
#PBS -l nodes=1:ppn=32:bigmemlong
#PBS -q bigmemlong
#PBS -l mem=80gb
#PBS -M YOUR_MAIL -m ae
#PBS -l walltime=RUNTIME
cd PATH_TO_YOUR_FOLDER
source ./21cm-wrapper-penv/bin/activate
python ./simulation.py
``` 
We save it as run_simulation.sh. <br />  <br />

Now can run it via <br />
```qsub run_simulation.sh```
. <br /> <br /> <br />

<b>MCMC (THIS IS STILL EXPERIMENTAL)</b> <br/><br/>
For affine-invariant ensemble sampling using emcee, <br/>
a starting script may look like this:<br/>
```python
# load class
from mcmc import *

# set first sampling range
init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [5, 100], 
                    "L_X": [30,50], "NU_X_THRESH": [100, 1500]}}

# define the log-prior
def log_prior(theta):
        T_vir, H_eff, LX, nu_x = theta
        if 5 < H_eff < 100 and 100 < nu_x < 1500 and 4 < T_vir < 5.3 and 30 < LX < 50:
            return 0
        return - np.inf

#define the log-likelihood
def log_likelihood(t,f):
    return - np.sum((t - f)**2/f)

# initialize sampler, choose nwalker twice the number of available cpu-cores
# for best performance
mcrun = mcmc(nwalker=8, z_bins = 11, debug = True,
	 log_likelihood=log_likelihood, prior=log_prior)
	 
# generate or load fiducial model
mcrun.make_fiducial(load = False)

# run the sampler
mcrun.run_aies(init_params_ranges)
```
The program can be queued as above. <br/> <br/>



For doing nested sampling via pymultinest, <br/>
a starting script may look like this:

```python
# load class
from mcmc import *

# set first sampling range
init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [5, 100], 
                    "L_X": [30,50], "NU_X_THRESH": [100, 1500]}}

# input: init cube for parameter [0,1)
# pymultinest samples parameter from the prior, not probabilities
def prior_sampling(theta):
        T_vir, H_eff, LX, nu_x = theta
        T_vir = 4 + 1.3*T_vir
        H_eff = 5 + 95*H_eff
        LX = 30 + 20*LX
        nu_x = 100 + 1400*nu_x
        return np.array([T_vir, H_eff, LX, nu_x])

# define the log-likelihood
def log_likelihood(t,f):
    return - np.sum((t - f)**2/f)

# initialize sampler
mcrun = mcmc(z_bins = 11, debug = True, log_likelihood=log_likelihood, prior=prior_sampling)

# generate or load fiducial model
mcrun.make_fiducial(load = True)

# run the sampler
mcrun.run_ns(init_params_ranges)
```
pymultinest requires multiprocessing via MPI <br/>
To run the script multithreaded, change the line in your run.sh to
```bash
mpiexec -n NUMBER-OF-CPU-CORES python main.py
```
<br/><br/>
