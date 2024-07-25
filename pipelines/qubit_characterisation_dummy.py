import numpy as np

from matplotlib import pyplot as plt

from scipy.signal import find_peaks

from helper_functions.data_access_layer import DataAccess
from base_stages_simple import BaseStage
from mock_data import noisy_lorentzian, noisy_rabi_oscillations, fit_rabi_oscillations, rabi_oscillations


class ResonanceFrequency(BaseStage):
    """
    A class for resonance frequency determination. It inherits from the BaseStage class.
    It includes functionality to prepare measurements, investigate a candidate, perform measurements, and determine candidates.
    """

    def __init__(
        self,
        experiment_name: str,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "ResFreq"
        super().__init__(experiment_name, configs, name, data_access_layer)


    def prepare_measurement(self):
        """Setting voltages, ramping magnets, preparing AWGs, etc."""
        pass


    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()

        # simulate measurement
        frequencies = np.linspace(1e9, 2e9, 1000)
        data = noisy_lorentzian(frequencies, 1.4e9, 1e7, 1, 1e-9)

        fig, axs = plt.subplots(1, 1)
        axs.scatter(frequencies, data)
        axs.set_title("frequency sweep")
        message = f"frequency sweep done"
        filename = f"frequency_sweep"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        self.data_access_layer.save_data({'frequency_sweep': {'frequencies': frequencies, 'data': data}})

        # You can save a GUID or a different pointer to the data
        self.current_candidate.data_identifiers["data"] = 'frequency_sweep'
        self.on_measurement_end()

    def determine_candidates(self):
        self.on_analysis_start()

        data_dict = self.data_access_layer.load_data(self.current_candidate.data_identifiers["data"])
        frequencies = data_dict['frequencies']
        data = data_dict['data']

        data_normed = (data - data.min()) / (data.max() - data.min())

        peaks, _ = find_peaks(data_normed, prominence=self.configs["prominence"])

        fig = plt.figure()
        plt.scatter(frequencies, data, label='data')
        plt.xlabel('frequency')
        plt.ylabel("response")
        plt.scatter(frequencies[peaks], data[peaks], marker='x', c='tab:red', s=100, label="peaks")

        filename = "frequency_sweep_with_peaks"

        message = f"frequency sweep with peaks, peaks at frequencies: {frequencies[peaks]}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        freq_of_interest = frequencies[peaks][0]

        # We need to create a list of dictionaries with the information of the candidates
        candidate_info = self.current_candidate.info
        candidate_info["data_freq_sweep"] = self.current_candidate.data_identifiers["data"]
        candidate_info["freq_of_interest"] = freq_of_interest
        candidates_info = [candidate_info]

        # on_analysis_end builds the candidates and returns them
        candidates = self.on_analysis_end(candidates_info)
        return candidates

class RabiOscillations(BaseStage):
    def __init__(
        self,
        experiment_name: str,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "RabiOscillations"
        super().__init__(experiment_name, configs, name, data_access_layer)

    def prepare_measurement(self):
        """Setting voltages, ramping magnets, preparing AWGs, etc."""
        pass

    def determine_candidates(self):
        self.on_analysis_start()

        data_dict = self.data_access_layer.load_data(self.current_candidate.data_identifiers["data"])
        wait_times = data_dict['wait_times']
        data = data_dict['data']

        t = wait_times
        A = 0.5
        omega = 5e8
        tau = 5e-8
        phase = 0

        signal = (data - np.min(data)) / (np.max(data) - np.min(data)) - 1/2
        guess = [A, omega, tau, phase]
        popt, pcov = fit_rabi_oscillations(t, signal, guess)
        omega_fitted = popt[1]
        rabi_frequency = omega_fitted / (2 * np.pi)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(t, signal, 'b-', label='data')
        plt.plot(t, rabi_oscillations(t, *popt), 'r-',)

        message = f'fit rabi oscillations \n' \
                  f'Fit results: A={popt[0]}, omega={popt[1]}, tau={popt[2]}, phase={popt[3]},\n' \
                  f'rabi_frequency {rabi_frequency}'

        plt.xlabel("wait times")
        plt.ylabel("response")
        fig_name = "rabi_oscillations_fit"
        self.data_access_layer.create_with_figure(message, fig, fig_name)
        plt.close()

        # We can plot data from previous stages together with current data:

        # load the data from the previous stage
        data_dict = self.data_access_layer.load_data(self.current_candidate.info["data_freq_sweep"])
        frequencies = data_dict['frequencies']
        data = data_dict['data']

        freq_picked = self.current_candidate.info["freq_of_interest"]
        data_at_freq = data[np.argmin(np.abs(frequencies - freq_picked))]

        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(frequencies, data)
        axs[0].axvline(x=freq_picked, linewidth=4, color='r', )
        axs[0].set_title("frequency sweep")
        axs[0].set_xlabel("frequency")
        axs[0].set_ylabel("response")
        axs[1].scatter(t, signal)
        axs[1].set_xlabel("wait times")
        axs[1].set_ylabel("response")
        axs[1].set_title("rabi oscillations")
        plt.tight_layout()
        message = f"Rabi oscillations and frequency sweep"
        filename = f"rabi_oscs_and_freq_sweep"
        self.data_access_layer.create_with_figure(message, fig, filename)
        # fig.close()

        candidates = self.on_analysis_end([])
        return candidates

    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()

        # simulate measurement
        wait_times = np.linspace(1e-9, 50e-9, 100)
        data = noisy_rabi_oscillations(wait_times, 1, 5e8, 5e-8, 0, 0.1)

        fig, axs = plt.subplots(1, 1)
        axs.scatter(wait_times, data)
        axs.set_title("Rabi oscillations")
        message = f"Rabi oscillations scan done"
        filename = f"rabi_oscs"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        self.data_access_layer.save_data({'rabi_osc': {'wait_times': wait_times, 'data': data}})

        # You can save a GUID or a different pointer to the data
        self.current_candidate.data_identifiers["data"] = 'rabi_osc'
        self.on_measurement_end()

