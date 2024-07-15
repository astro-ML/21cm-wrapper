from Leaf import *
from powerbox.tools import get_power
import emcee
import dynesty


class KeyMismatchError(Exception):
    pass


class Probability:
    def __init__(
        self,
        prior_ranges: dict,
        z_chunks: int | list[int],
        bins: int,
        debug: bool = True,
        fmodel_path: str = "./mcmc_data/fiducial_cone.npy",
    ):
        """Stores the likelihood, priors and the summary statistics"""
        self.dodebug = debug
        self.chunks = z_chunks
        self.bins = bins
        self.fmodel_path = fmodel_path
        self.prior_ranges = prior_ranges

        def replace_lists_with_zero(d):
            if isinstance(d, dict):
                return {k: replace_lists_with_zero(v) for k, v in d.items()}
            elif isinstance(d, list):
                return 0
            else:
                return d

        self.parameter = replace_lists_with_zero(prior_ranges.copy())

    def log_probability(self, parameters, lightcone):
        prob = np.log(self.likelihood(lightcone=lightcone)) + self.log_prior_emcee(parameters=parameters)
        self.debug(f"Probability is {prob}")
        return prob

    def likelihood(self, lightcone: object):
        """Likelihood has lightcone object as input and outputs the likelihood"""
        fid_ps = np.load(self.fmodel_path)[:, 0, :]
        test_ps = self.ps1d(lightcone=lightcone)[:, 0, :]
        chi2 = self.loss(test_lc=test_ps, fiducial_lc=fid_ps)
        self.debug(f"Likelihood={chi2}")
        return chi2

    # def prior_ns(self, parameter: list):

    def prior_emcee(self, parameters: dict):
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
        res = self.prior_emcee(parameters)
        return 0 if res else - np.inf

    def loss(self, test_lc, fiducial_lc):
        loss = np.sum( (test_lc - fiducial_lc)**2 / (np.abs(fiducial_lc) + 1))
        return - loss

    def ps1d(self, lightcone: object):
        """Compute 1D PS"""
        if type(self.chunks) == int:
            self.debug("Compute 1D PS using int chunks")
            zbins = np.linspace(
                0, lightcone.lightcone_redshifts.shape[0] - 1, self.chunks + 1
            ).astype(int)
        else:
            self.debug("Compute 1D PS using list chunks")
            zbins = self.chunks
        ps = np.empty((len(zbins), 2, self.bins))
        for bin in range(len(zbins) - 1):
            # get variance=False for now until nice usecase is found
            ps[bin, :, :] = get_power(
                deltax=lightcone.brightness_temp[:, :, zbins[bin] : zbins[bin + 1]],
                boxlength=lightcone.cell_size
                * np.array(
                    [*lightcone.brightness_temp.shape[:2], zbins[bin + 1] - zbins[bin]]
                ),
                bin_ave=True,
                ignore_zero_mode=True,
                get_variance=False,
                bins=self.bins,
                vol_normalised_power=True,
            )
            ps[bin, 0, :] *= ps[bin, 1, :] ** 3 / (2 * np.pi**2)
            self.debug(
                f"PS is {ps[bin,0,:]}" + f" for bin {bin}" + f"\nfor ks {ps[bin,1,:]}"
            )
        return ps

    def debug(self, msg):
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
        """Stores everything related to the simulation like the fiducical model or noise"""
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
            self.fiducial_ps = Probability.ps1d(fiducial_cone)
            np.save(data_path + "fiducial_ps.npy", self.fiducial_ps)
            if debug:
                print("PS is ", self.fiducial_ps)
            if debug:
                print("New summary statistics successfully computed and saved.")

    def step(self, parameters: list[float]) -> float:
        # convert parameter list to dict
        parameters = Simulation.replace_values(self.Probability.parameter, parameters)
        self.debug("Current parameters are:" + str(parameters))
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
        return data + np.random.normal(mu, sigma, data.shape)

    @staticmethod
    def replace_values(nested_dict, values_array):
        values_array = list(values_array)
        """
        Replace all values in the nested dictionary with the elements of the values_array in order.
        
        :param nested_dict: A dictionary which may contain other dictionaries as values.
        :param values_array: A list of values to replace in the nested dictionary.
        :return: A new dictionary with replaced values.
        :raises ValueError: If the number of values in the dictionary does not match the length of values_array.
        """

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
        noise_type: None - no noise; syntax: (noise_type, **kwargs for noise)
                    list of noise_tupes:
                    (1, mu, sig) for guassian noise
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
        self.debug("Starting emcee sampling...")
        ndim = self.num_elements(self.Prob.prior_ranges)
        self.debug("Number of parameters: " + str(ndim))
        backend = emcee.backends.HDFBackend(filename=filename)
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
        initial_params = np.empty((shape))
        # print(self.Prob.prior_ranges, self.uniform)
        for i in range(shape[0]):
            initial_params[i, :] = self.extract_values(
                self.generate_range(self.Prob.prior_ranges, self.uniform)
            )
        return initial_params
