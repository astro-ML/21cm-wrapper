from Leaf import *
from py21cmfast_tools import calculate_ps
from powerbox.tools import ignore_zero_absk
import emcee
import dynesty
my_module_path = os.path.join("../", '21cm-sbi')
sys.path.append(my_module_path)
from dataloader import *


class Probability:
    def __init__(
        self,
        prior_ranges: dict,
        z_eval: int | list[int] = np.linspace(5.5, 25, 10),
        bins: int = 10,
        debug: bool = True,
        fmodel_path: str = "./mcmc_data/fiducial_ps.npy",
        summary_statistics: str = "1dps",
        summary_net = None,
        z_cut: int = 680,
    ):
        """Stores the likelihood, priors and the summary statistics

        Args:
            prior_ranges (dict): A dictionary containing the prior ranges for the parameters.
            z_eval (list[float]): A list of redshifts at which the power spectrum will be evaluated.
            bins (int): The number of bins for the power spectrum.
            debug (bool, optional): Whether to enable debug mode. Defaults to True.
            fmodel_path (str, optional): The path to the fiducial model. Defaults to "./mcmc_data/fiducial_cone.npy".
            summary_statistics (str, optional): The type of summary statistics to compute. Defaults to "1dps".
            summary_net (nn.Module, optional): The neural network for computing the summary statistics. Defaults to None.
            z_cut (int, optional): The redshift cut-off, only required if summary_net is not None to normalize size of lightcones. Defaults to 600.
        """
        self.dodebug = debug
        self.z_eval = z_eval
        self.bins = bins
        self.fmodel_path = fmodel_path
        self.prior_ranges = prior_ranges
        self.sum_stat = summary_statistics
        self.z_cut = 680
        
        if summary_net is None:
            self.sum_net = False
        else:
            self.sum_net = True
            self.summary_model = summary_net

        def replace_lists_with_zero(d):
            if isinstance(d, dict):
                return {k: replace_lists_with_zero(v) for k, v in d.items()}
            elif isinstance(d, list):
                return 0
            else:
                return d

        self.parameter = replace_lists_with_zero(prior_ranges.copy())

    def summary_statistics(self, lightcone: object):
        """Compute the summary statistics based on the specified type.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            object: The computed summary statistics.
        """
        match self.sum_stat:
            case "1dps":
                return self.ps1d(lightcone)
            case "2dps":
                return self.ps2d(lightcone)
            case "summary_net":
                return self.summary_net(lightcone)
            case _:
                print("Summary statistics not found")

    def log_probability(self, parameters, lightcone):
        """Compute the log probability.

        Args:
            parameters: The parameters.
            lightcone: The lightcone object.

        Returns:
            float: The log probability.
        """
        prob = self.likelihood(lightcone=lightcone) + self.log_prior_emcee(parameters=parameters)
        self.debug(f"Probability is {prob}")
        return prob

    def likelihood(self, lightcone: object):
        """Compute the likelihood.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            float: The likelihood.
        """
        fid_ps = np.load(self.fmodel_path)
        fid_ps, variance = fid_ps
        test_ps, _ = self.summary_statistics(lightcone=lightcone)
        chi2 = self.loss(test_lc=test_ps, fiducial_lc=fid_ps, var=variance)
        self.debug(f"Likelihood={chi2}")
        self.debug(f"LogLikelihood={np.log(-chi2)}")
        return chi2

    # def prior_ns(self, parameter: list):

    def prior_emcee(self, parameters: dict):
        """Compute the prior probability using the emcee sampler.

        Args:
            parameters (dict): The parameters.

        Returns:
            int: The prior probability.
        """
        dict1 = parameters
        dict2 = self.prior_ranges

        def compare_values(val1, range_tuple):
            return range_tuple[0] <= val1 <= range_tuple[1]

        def compare_nested(dict1, dict2):
            for key, val in dict1.items():
                if key not in dict2:
                    raise KeyMismatchError(
                        f"Key '{key}' found in dict1 but not in dict2"
                    )
                if isinstance(val, dict):
                    if not isinstance(dict2[key], dict):
                        raise KeyMismatchError(
                            f"Value for key '{key}' is a dict in dict1 but not in dict2"
                        )
                    if not compare_nested(val, dict2[key]):
                        return 0
                else:
                    if not compare_values(val, dict2[key]):
                        return 0
            return 1

        result = compare_nested(dict1, dict2)
        self.debug(f"Prior is {result}")
        return result

    def log_prior_emcee(self, parameters: dict):
        """Compute the log prior probability using the emcee sampler.

        Args:
            parameters (dict): The parameters.

        Returns:
            float: The log prior probability.
        """
        res = self.prior_emcee(parameters)
        return 0 if res else - np.inf

    def loss(self, test_lc, fiducial_lc, var):
        """Compute the loss function.
            shape must be [bins, [data, variance], *data] = (bins, 2, *data)
            We also assume implicit Gaussian prior, which for large data isn't
            a bad start

        Args:
            test_lc: The test lightcone.
            fiducial_lc: The fiducial lightcone.
            var: variance of the test lightcone

        Returns:
            float: The loss value.
        """
        print("computing loss")
        sig = np.sqrt(var) + 1 # np.sqrt(fiducial_lc) + np.sqrt(test_lc) + 1e-5
        loss = - 0.5*np.sum( (test_lc - fiducial_lc)**2 
                            / sig
                            + np.log(sig))
        return loss

    def prior_dynasty(self, parameters: NDArray) -> NDArray:
        """Compute the prior probability using the Dynasty sampler.

        Args:
            parameters (NDArray): The parameters.

        Returns:
            NDArray: The prior probability.
        """
        parameter_ranges = np.array(extract_values(self.prior_ranges))
        parameters *= np.diff(parameter_ranges)[:,0]
        parameters += parameter_ranges[:,0] 
        self.debug("Prior: " + str(parameters))
        return parameters
    
    def summary_net(self, lightcone: object) -> NDArray:
        """Compute the summary statistics using the neural network.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            NDArray: The computed summary statistics.
        """
        bt = np.expand_dims(np.expand_dims(np.array(lightcone.lightcones['brightness_temp'][:,:,:self.z_cut]), 0),0)
        diff = bt.max() - bt.min()
            # normalize to [0,1]
        bt = (bt - bt.min()) / diff
        return self.summary_model(torch.as_tensor(bt)).detach().numpy(), 1

    def ps1d(self, lightcone: object) -> NDArray:
        """Compute the 1D power spectrum.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            NDArray: The computed 1D power spectrum.
        """

        res = calculate_ps(lc = lightcone.lightcones['brightness_temp'] , 
                           lc_redshifts=lightcone.lightcone_redshifts, 
                           box_length=lightcone.user_params.BOX_LEN, 
                           box_side_shape=lightcone.user_params.HII_DIM,
                           log_bins=False, zs = self.z_eval, 
                           calc_1d=True, calc_2d=False, get_variance=True,
                           nbins_1d=self.bins, bin_ave=True, 
                           k_weights=ignore_zero_absk,postprocess=True)
        return res['ps_1D'], res['var_1D']

    def ps2d(self, lightcone: object) -> NDArray:
        """Compute the 2D power spectrum.

        Args:
            data: The data.
            size: The size of the data.

        Returns:
            NDArray: The computed 2D power spectrum.
        """
        
        res = calculate_ps(lc = lightcone.lightcones['brightness_temp'] , lc_redshifts=lightcone.lightcone_redshifts, 
                           box_length=lightcone.user_params.BOX_LEN, box_side_shape=lightcone.user_params.HII_DIM,
                           log_bins=False, zs = self.z_eval, calc_1d=False, calc_2d=True, get_variance=True,
                           nbins=self.bins, bin_ave=True, k_weights=ignore_zero_absk, postprocess=True)
        return res['final_ps_2D'], res['full_var_2D']

    def debug(self, msg):
        """Print the debug message.

        Args:
            msg: The debug message.
        """
        if self.dodebug:
            print(msg)


class Simulation(Leaf):
    def __init__(
        self,
        Probability: Probability,
        redshift: float,
        data_path: str = "./mcmc_data/",
        noise_type: tuple = None,
        debug: bool = False,
        regenerate_fiducial: bool = True,
        **fid_params,
    ):
        """
        Initializes a Simulation object.

        Args:
            Probability (Probability): The probability object used for computing the log probability.
            redshift (float): The redshift value for the simulation.
            data_path (str, optional): The path to the data directory. Defaults to "./mcmc_data/".
            noise_type (tuple, optional): The type of noise to apply to the simulation. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            regenerate_fiducial (bool, optional): Whether to regenerate the fiducial lightcone. Defaults to True.
            **fid_params: Additional parameters for the fiducial model.

        Raises:
            ValueError: If the number of values in the dictionary does not match the length of values_array.
        """
        # initialize lightcones with fid_params
        super().__init__(data_path=data_path, debug=debug, **fid_params)
        self.debug("initialize Simulation class...")
        self.redshift = redshift
        self.Probability = Probability
        self.noise = noise_type
        self.fid_file = fnmatch.filter(os.listdir(self.data_path), "fiducial_cone.h5")
        self.ps_file = fnmatch.filter(os.listdir(self.data_path), "fiducial_ps.npy")
        self.regenerate_fiducial = regenerate_fiducial
        self.data_path = data_path
        
        
    def make_fiducial(self, random_seed: int = None):

        self.debug("Search for existing fiducial lightcone...")
        
        if (len(self.fid_file) != 0) and not self.regenerate_fiducial:
            fiducial_cone = self.load(
            self.data_path + "fiducial_cone.h5", lightcone=True
            )

            self.debug("Existing lightcone successfully loaded.")
            
        else:
            if self.regenerate_fiducial and (len(self.fid_file) != 0):
                os.remove(self.data_path + "fiducial_cone.h5")
            
            temp_threads = self.userparams.N_THREADS
            self.userparams.update(
            N_THREADS=os.cpu_count() if os.cpu_count() < 33 else 32
            )
            fiducial_cone = self.run_lightcone(
            redshift=self.redshift,
            save=False,
            # fixed see because fiducial lightcones should look the same
            random_seed=random_seed,
            filter_peculiar=False,
            sanity_check=True,
            )
            self.userparams.update(N_THREADS=temp_threads)
            if self.noise is not None:
                if self.noise[0] == 1:
                    self.debug("Gaussian noise will be added...")
                    fiducial_cone.lightcones['brightness_temp']  = Simulation.gaussian_noise(
                        fiducial_cone.lightcones['brightness_temp'] , *self.noise[1:]
                    )
                elif self.noise[0] == 2:
                    print("Better noise model here")
                    # test_lc.lightcones['brightness_temp']  = Simulation.gaussian_noise(test_lc.lightcones['brightness_temp'] , **self.noise[1:])
                else:
                    print("Noise-type not found you gave: ", self.noise)
            self.save(
            obj=fiducial_cone, fname="fiducial_cone", direc=self.data_path, run_id=""
            )

            self.debug("New lightcone successfully computed and saved.")

        if np.isnan(fiducial_cone.lightcones['brightness_temp'] ).any(): 
            raise ValueError("Brightness temperature contains NaNs!" + 
                                "Please check your parameters and try again")

        self.debug("Search for existing summary statistics...")
        if (len(self.ps_file) != 0) and not self.regenerate_fiducial:
            self.fiducial_ps = np.load(self.data_path + "fiducial_ps.npy")

            self.debug("Existing summary statistics successfully loaded.")

            self.debug("PS is " + str(self.fiducial_ps))
        else:
            if self.regenerate_fiducial and (len(self.ps_file) != 0):
                os.remove(self.data_path + "fiducial_ps.npy")

            # 1dps hardcoded change to generic summary statistic in the future
            self.fiducial_ps = self.Probability.summary_statistics(fiducial_cone)
            np.save(self.data_path + "fiducial_ps.npy", self.fiducial_ps)

            self.debug("PS is " + str(self.fiducial_ps))
            self.debug("New summary statistics successfully computed and saved.")
        if np.isnan(self.fiducial_ps).any(): 
            raise ValueError("Summary statistics contains NaNs!" + 
                                "Please check your parameters and try again")

    def step(self, parameters: list[float]) -> float:
        """
        Performs a simulation step.

        Args:
            parameters (list[float]): The list of parameters for the simulation.

        Returns:
            float: The log probability of the simulation.

        Raises:
            ValueError: If the number of values in the dictionary does not match the length of values_array.
        """
        # convert parameter list to dict
        parameters = Simulation.replace_values(self.Probability.parameter, parameters)
        self.debug("Current parameters are:" + str(parameters))
        if not self.Probability.prior_emcee(parameters):
            self.debug("Return -inf due to off-boundary prior")
            return - np.inf
        # run lightcone sim
        test_lc = self.run_lightcone(
            redshift=self.redshift,
            **parameters,
            sanity_check=True,
            filter_peculiar=False,
            save=False,
        )
        if self.noise is not None:
            if self.noise[0] == 1:
                test_lc.lightcones['brightness_temp']  = Simulation.gaussian_noise(
                    test_lc.lightcones['brightness_temp'] , *self.noise[1:]
                )
            elif self.noise[0] == 2:
                print("Better noise model here")
                # test_lc.lightcones['brightness_temp']  = Simulation.gaussian_noise(test_lc.lightcones['brightness_temp'] , **self.noise[1:])
            else:
                print("Noise-type not found you gave: ", self.noise)
        # compute probability
        return self.Probability.log_probability(
            lightcone=test_lc, parameters=parameters
        )

    @staticmethod
    def gaussian_noise(data: NDArray, mu: float, sigma: float) -> NDArray:
        """
        Adds Gaussian noise to the data.

        Args:
            data (NDArray): The input data.
            mu (float): The mean of the Gaussian distribution.
            sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
            NDArray: The data with added Gaussian noise.
        """
        return data + np.random.normal(mu, sigma, data.shape)

    @staticmethod
    def replace_values(nested_dict, values_array):
        """
        Replace all values in the nested dictionary with the elements of the values_array in order.

        Args:
            nested_dict: A dictionary which may contain other dictionaries as values.
            values_array: A list of values to replace in the nested dictionary.

        Returns:
            A new dictionary with replaced values.

        Raises:
            ValueError: If the number of values in the dictionary does not match the length of values_array.
        """
        values_array = list(values_array)

        def count_values(d):
            """Recursively count the total number of values in the nested dictionary."""
            count = 0
            for value in d.values():
                if isinstance(value, dict):
                    count += count_values(value)
                else:
                    count += 1
            return count

        def replace(d, values):
            """Recursively replace values in the nested dictionary with values from the list."""
            for key in d.keys():
                if isinstance(d[key], dict):
                    replace(d[key], values)
                else:
                    d[key] = values.pop(0)

        total_values = count_values(nested_dict)

        if total_values != len(values_array):
            raise ValueError(
                "The number of values in the dictionary does not match the length of the values array."
            )
        replace(nested_dict, values_array)

        return nested_dict


class Flower(Simulation):
    def __init__(
        self,
        Probability: Probability,
        data_path: str = "./mcmc_data/",
        noise_type: tuple = None,
        debug: bool = False,
        regenerate_fiducial: bool = True,
        redshift: float = 5.5,
        fid_params: dict = {},
    ):
        """
        A class representing a flower simulation.

        Args:
            Probability (Probability): An instance of the Probability class.
            data_path (str, optional): The path to the data directory. Defaults to "./mcmc_data/".
            noise_type (tuple, optional): The type of noise. Defaults to None.
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
            regenerate_fiducial (bool, optional): Flag to regenerate fiducial parameters. Defaults to True.
            redshift (float, optional): The redshift value. Defaults to 5.5.
            fid_params (dict, optional): Additional fiducial parameters. Defaults to {}.
        """
        super().__init__(
            Probability=Probability,
            redshift=redshift,
            data_path=data_path,
            noise_type=noise_type,
            debug=debug,
            regenerate_fiducial=regenerate_fiducial,
            **fid_params,
        )
        self.Prob = Probability

    def run_emcee(
        self,
        filename: str = "./results_emcee.h5",
        threads: int = 1,
        walkers: int = 12,
        nsteps: int = 1000,
    ) -> None:
        """
        Run the emcee sampling.

        Args:
            filename (str, optional): The filename to save the results. Defaults to "./results_emcee.h5".
            threads (int, optional): The number of threads to use. Defaults to 1.
            walkers (int, optional): The number of walkers. Defaults to 12.
            nsteps (int, optional): The number of steps. Defaults to 1000.
        """
        self.debug("Starting emcee sampling...")
        ndim = num_elements(self.Prob.prior_ranges)
        self.debug("Number of parameters: " + str(ndim))
        backend = emcee.backends.HDFBackend(filename=self.data_path + filename)
        initial = self.initialize_parameter((walkers, ndim))
        self.debug("Initial parameters: " + str(initial))
        schwimmhalle = Pool(
            max_workers=threads, max_tasks_per_child=1, mp_context=get_context("spawn")
        )
        with schwimmhalle as p:
            sampler = emcee.EnsembleSampler(
                nwalkers=walkers,
                ndim=ndim,
                log_prob_fn=self.step,
                pool=p,
                backend=backend,
            )
            sampler.run_mcmc(initial_state=initial, nsteps=nsteps, progress=True)

    def initialize_parameter(self, shape: tuple[int, int]):
        """
        Initialize the parameters.

        Args:
            shape (tuple[int, int]): The shape of the parameters.

        Returns:
            np.ndarray: The initialized parameters.
        """
        initial_params = np.empty((shape))
        # print(self.Prob.prior_ranges, self.uniform)
        for i in range(shape[0]):
            initial_params[i, :] = extract_values(
                generate_range(self.Prob.prior_ranges, self.uniform)
            )
        return initial_params
    
    def run_ns(self, filename: str = "./results_dynasty", threads: int = 1, npoints: int = 250, **dynasty_params):
        """
        Run the nested sampling.

        Args:
            filename (str, optional): The filename to save the results. Defaults to "./results_dynasty".
            threads (int, optional): The number of threads to use. Defaults to 1.
            npoints (int, optional): The number of live points. Defaults to 250.
            **dynasty_params: Additional parameters for the nested sampling algorithm.
        """
        self.debug("Starting nested sampling...")
        ndim = num_elements(self.Prob.prior_ranges)
        self.debug("Number of parameters: " + str(ndim))
        schwimmhalle = Pool(
            max_workers=threads, max_tasks_per_child=1, mp_context=get_context("spawn")
        )
        with schwimmhalle as p:
            sampler = dynesty.NestedSampler(loglikelihood=self.step, 
                                        prior_transform=self.Probability.prior_dynasty, 
                                        ndim=ndim, nlive = npoints, bound='cubes', # 'multi'
                                        pool=p, queue_size = threads, sample='rwalk',)
                                        #first_update={'min_ncall': npoints, 'min_eff': 20.}) 
            sampler.run_nested(checkpoint_file=self.data_path + filename, **dynasty_params)

    def run_dns(self, filename: str = "./results_dynasty", threads: int = 1, npoints: int = 250, **dynasty_params):
        """
        Run the nested sampling.

        Args:
            filename (str, optional): The filename to save the results. Defaults to "./results_dynasty".
            threads (int, optional): The number of threads to use. Defaults to 1.
            npoints (int, optional): The number of live points. Defaults to 250.
            **dynasty_params: Additional parameters for the nested sampling algorithm.
        """
        self.debug("Starting nested sampling...")
        ndim = num_elements(self.Prob.prior_ranges)
        print("ndim =", ndim)
        self.debug("Number of parameters: " + str(ndim))
        schwimmhalle = Pool(
            max_workers=threads, max_tasks_per_child=1, mp_context=get_context("spawn")
        )
        with schwimmhalle as p:
            dsampler = dynesty.DynamicNestedSampler(loglikelihood=self.step, 
                                        prior_transform=self.Probability.prior_dynasty, 
                                        ndim=ndim, bound='multi', # 'multi'
                                        pool=p, queue_size = threads, sample='auto',)
                                        #first_update={'min_ncall': npoints, 'min_eff': 20.}) 
            dsampler.run_nested(checkpoint_file=self.data_path + filename, **dynasty_params, print_progress=True)
            


        
