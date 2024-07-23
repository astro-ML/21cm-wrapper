from Leaf import *
from powerbox.tools import get_power
import emcee
import dynesty
my_module_path = os.path.join("./", '21cm-PS')
sys.path.append(my_module_path)
from power_spectra import PowerSpectrum
from data_processing import DataProcessor


class KeyMismatchError(Exception):
    pass


class Probability:
    def __init__(
        self,
        prior_ranges: dict,
        z_chunks: int | list[int],
        bins: int,
        debug: bool = True,
        fmodel_path: str = "./mcmc_data/fiducial_ps.npy",
        summary_statistics: str = "1dps"
    ):
        """Stores the likelihood, priors and the summary statistics

        Args:
            prior_ranges (dict): A dictionary containing the prior ranges for the parameters.
            z_chunks (int | list[int]): The number of chunks or a list of chunk indices for computing the power spectrum.
            bins (int): The number of bins for the power spectrum.
            debug (bool, optional): Whether to enable debug mode. Defaults to True.
            fmodel_path (str, optional): The path to the fiducial model. Defaults to "./mcmc_data/fiducial_cone.npy".
            summary_statistics (str, optional): The type of summary statistics to compute. Defaults to "1dps".
        """
        self.dodebug = debug
        self.chunks = z_chunks
        self.bins = bins
        self.fmodel_path = fmodel_path
        self.prior_ranges = prior_ranges
        self.ps = PowerSpectrum()
        self.ps.data = DataProcessor()
        self.sum_stat = summary_statistics

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
        test_ps = self.summary_statistics(lightcone=lightcone)
        chi2 = self.loss(test_lc=test_ps, fiducial_lc=fid_ps)
        self.debug(f"Likelihood={chi2}")
        self.debug(f"LogLikelihood={np.log(chi2)}")
        return - chi2

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

    def loss(self, test_lc, fiducial_lc):
        """Compute the loss function.
            shape must be [bins, [data, variance], *data] = (bins, 2, *data)

        Args:
            test_lc: The test lightcone.
            fiducial_lc: The fiducial lightcone.

        Returns:
            float: The loss value.
        """
        print("computing loss")
        loss = - 0.5*np.sum( (test_lc[:,0] - fiducial_lc[:,0])**2 / (test_lc[:,1]**2 + fiducial_lc[:,1]**2) + np.log(test_lc[:,1]**2 + fiducial_lc[:,1]**2))
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

    def ps1d(self, lightcone: object):
        """Compute the 1D power spectrum.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            object: The computed 1D power spectrum.
        """
        if type(self.chunks) == int:
            self.debug("Compute 1D PS using int chunks")
            zbins = np.linspace(
                0, lightcone.lightcone_redshifts.shape[0] - 1, self.chunks + 1
            ).astype(int)
        else:
            self.debug("Compute 1D PS using list chunks")
            zbins = self.chunks
        if type(self.bins) == int:
            ps = np.empty((len(zbins)-1, 2, self.bins))
        else:
            ps = np.empty((len(zbins)-1, 2, len(self.bins)-1))
        for bin in range(len(zbins) - 1):
            # get variance=False for now until nice usecase is found
            ps[bin, 0, :], k, ps[bin, 1, :] = get_power(
                deltax=lightcone.brightness_temp[:, :, zbins[bin] : zbins[bin + 1]],
                boxlength=lightcone.cell_size
                * np.asarray(lightcone.brightness_temp.shape),
                bin_ave=True,
                ignore_zero_mode=True,
                get_variance=True,
                bins=self.bins,
                vol_normalised_power=True,
            )
            ps[bin, :, :] *= k ** 3 
            self.debug(
                f"PS is {ps[bin,0,:]}" + f" for bin {bin}" + f"\nfor ks {k}"
            )
        return ps

    def ps2d(self, lightcone: object):
        """Compute the 2D power spectrum.

        Args:
            lightcone (object): The lightcone object.

        Returns:
            object: The computed 2D power spectrum.
        """
        if type(self.chunks) == int:
            self.debug("Compute 2D PS using int chunks")
            zbins = np.linspace(
                0, lightcone.lightcone_redshifts.shape[0] - 1, self.chunks + 1
            ).astype(int)
        else:
            self.debug("Compute 2D PS using list chunks")
            zbins = self.chunks
        if type(self.bins) == int:
            ps = np.empty((len(zbins)-1, 2, self.bins, self.bins))
        else:
            ps = np.empty((len(zbins)-1, 2, len(self.bins), len(self.bins)))
        for bin in range(len(zbins) - 1):
            # get variance=False for now until nice usecase is found
            field = lightcone.brightness_temp[:,:,zbins[bin]:zbins[bin+1]]
            k_perp, k_par, ps[bin,0, :, :] = self.compute_ps2d(field, lightcone.cell_size*field.shape)
            self.debug(
                f"PS is {ps[bin,0,:,:]}" + f" in {bin} for k_perp {k_perp}" + f"\nfor k_par {k_par}"
            )
        return ps

    def compute_ps2d(self, lightcone: object):
        """Compute the 2D power spectrum.

        Args:
            data: The data.
            size: The size of the data.

        Returns:
            tuple: The k_perp, k_par, and the computed 2D power spectrum.
        """
        
        if type(self.chunks) == int:
            self.debug("Compute 2D PS using int chunks")
            zbins = np.linspace(
                0, lightcone.lightcone_redshifts.shape[0] - 1, self.chunks + 1
            ).astype(int)
        else:
            self.debug("Compute 2D PS using list chunks")
            zbins = self.chunks
        if type(self.bins) == int:
            ps = np.empty((len(zbins)-1, 2, self.bins, self.bins))
        else:
            ps = np.empty((len(zbins)-1, 2, len(self.bins), len(self.bins)))
        for bin in range(len(zbins) - 1):
            # get variance=False for now until nice usecase is found
            field = lightcone.brightness_temp[:,:,zbins[bin]:zbins[bin+1]]
            
            ps_perp,k_perp, _, var_perp = get_power(field, boxlength=lightcone.cell_size*np.asarray(field.shape), res_ndim=2, bins = self.bins, 
                                    ignore_zero_mode=False, bin_ave=True, get_variance=True) 
            ps_par, k_par, var_par = get_power(field.T, boxlength=lightcone.cell_size*np.asarray(field.shape), res_ndim=1, bins = self.bins, 
                                        ignore_zero_mode=False, bin_ave=True, get_variance=True)
            ps_perp = np.mean(ps_perp,axis=1)
            ps_par = np.mean(ps_par, axis=(1,2))
            ps[bin,0,:,:] = np.outer(ps_perp*k_perp**2, ps_par*k_par).T
            ps[bin,1,:,:] = np.outer(var_perp*k_perp**2, var_par*k_par).T
            self.debug(
                f"PS is {ps[bin,0,:,:]}" + f" in {bin} for k_perp {k_perp}" + f"\nfor k_par {k_par}"
            )
        return ps

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
        fid_file = fnmatch.filter(os.listdir(self.data_path), "fiducial_cone.h5")
        ps_file = fnmatch.filter(os.listdir(self.data_path), "fiducial_ps.npy")
        if debug:
            print("Search for existing fiducial lightcone...")
        if (len(fid_file) != 0) and not regenerate_fiducial:

            fiducial_cone = self.load(
                self.data_path + "fiducial_cone.h5", lightcone=True
            )
            if debug:
                print("Existing lightcone successfully loaded.")
        else:
            if regenerate_fiducial and (len(fid_file) != 0):
                os.remove(data_path + "fiducial_cone.h5")
            temp_threads = self.userparams.N_THREADS
            self.userparams.update(
                N_THREADS=os.cpu_count() if os.cpu_count() < 33 else 32
            )
            fiducial_cone = self.run_lightcone(
                redshift=redshift,
                save=False,
                # fixed see because fiducial lightcones should look the same
                random_seed=42,
                filter_peculiar=False,
                sanity_check=True,
            )
            self.userparams.update(N_THREADS=temp_threads)
            self.save(
                obj=fiducial_cone, fname="fiducial_cone", direc=data_path, run_id=""
            )
            if debug:
                print("New lightcone successfully computed and saved.")

        if debug:
            print("Search for existing summary statistics...")
        if (len(ps_file) != 0) and not regenerate_fiducial:
            self.fiducial_ps = np.load(self.data_path + "fiducial_ps.npy")
            if debug:
                print("Existing summary statistics successfully loaded.")
            if debug:
                print("PS is ", self.fiducial_ps)
        else:
            if regenerate_fiducial and (len(ps_file) != 0):
                os.remove(data_path + "fiducial_ps.npy")

            # 1dps hardcoded change to generic summary statistic in the future
            self.fiducial_ps = self.Probability.summary_statistics(fiducial_cone)
            np.save(data_path + "fiducial_ps.npy", self.fiducial_ps)
            if debug:
                print("PS is ", self.fiducial_ps)
            if debug:
                print("New summary statistics successfully computed and saved.")

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
                test_lc.brightness_temp = Simulation.gaussian_noise(
                    test_lc.brightness_temp, **self.noise[1:]
                )
            elif self.noise[0] == 2:
                print("Better noise model here")
                # test_lc.brightness_temp = Simulation.gaussian_noise(test_lc.brightness_temp, **self.noise[1:])
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
                                        ndim=ndim, nlive = npoints, bound='balls',
                                        pool=p, queue_size = threads) 
            sampler.run_nested(dlogz=0.5, checkpoint_file=self.data_path + filename)

        
