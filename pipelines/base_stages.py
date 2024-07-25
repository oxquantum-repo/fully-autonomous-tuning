import hashlib
import json
import logging
import os
import pickle
import time
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Callable, Union

import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.stats import qmc
from treelib import Tree
from treelib.exceptions import DuplicatedNodeIdError

from helper_functions.data_access_layer import DataAccess
from pipelines.investigation_result import InvestigationResult
from pipelines.utils import get_current_timestamp


@dataclass
class Candidate:
    sender: str
    receiver: str
    sender_candidate_id: int
    name: str
    info: Dict[str, Union[float, bool, str, dict]]
    data_taken: bool = False
    analysis_done: bool = False
    times: Dict = field(default_factory=dict)
    data_identifiers: Dict = field(default_factory=dict) # store link to previously taken data here, either a GUID or a name that can be loaded via DataAccess
    resulting_candidates: List = field(default_factory=list)

    def __str__(self):
        '''For prettier prints'''
        return (f"Candidate(\n"
                f"\tsender={self.sender},\n"
                f"\treceiver={self.receiver},\n"
                f"\tsender_candidate_id={self.sender_candidate_id},\n"
                f"\tname={self.name},\n"
                f"\tinfo={pformat(self.info)},\n"
                f"\tdata_taken={self.data_taken},\n"
                f"\tanalysis_done={self.analysis_done},\n"
                f"\ttimes={self.times},\n"
                f"\tdata_identifiers={pformat(self.data_identifiers)},\n"
                f"\tresulting_candidates={pformat(self.resulting_candidates)}\n"
                f")")

@dataclass
class StageState:
    """Used to communicate states"""

    received_candidates: List[Candidate] = field(default_factory=list)
    current_candidate_idx: int = None
    candidate_names: List[str] = field(default_factory=list)


class BaseStage:
    """Base class for tuning stages"""

    def __init__(
        self,
        experiment_name: str,
        configs: dict,
        name: str,
        data_access_layer: DataAccess,
    ):
        """
        Initialise parameters that all stages have in common.

        Args:
            experiment_name (str): Name of the experiment.
            configs (dict): Configuration parameters for the tuning stage.
            name (str): Name of this stage.
            data_access_layer (DataAccess): An instance to access data.
        """
        self.experiment_name = experiment_name
        self.configs = configs
        self.name = name
        self.data_access_layer = data_access_layer
        self.state = StageState()
        self.parent_stages: List[BaseStage] = []
        self.child_stages: List[BaseStage] = []
        self.current_candidate = None
        self.show_tree = False

    def prepare_measurement(self):
        """Prepare the measurements for the current stage. To be implemented in subclasses."""
        pass

    def add_candidate(self, candidate):
        """
        Add a new candidate to the stage.

        Args:
            candidate (Candidate): Candidate to be added.
        """
        new_name = candidate.name
        if new_name in self.state.candidate_names:
            self.state.current_candidate_idx = self.state.candidate_names.index(
                new_name
            )
        else:
            self.state.candidate_names.append(new_name)
            self.state.received_candidates.append(candidate)
            self.state.current_candidate_idx = len(self.state.received_candidates) - 1
        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]

    def update_candidate(self, candidate: Candidate):
        """
        Update an existing candidate.

        Args:
            candidate (Candidate): Candidate to be updated.
        """
        self.state.current_candidate_idx = self.state.candidate_names.index(
            candidate.name
        )
        self.state.received_candidates[self.state.current_candidate_idx] = candidate
        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]



    def save_timestamp(self, message: str = ''):
        """Save the state of the current stage."""
        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        node_name = self.name + "_" + current_name + ".txt"
        file_name = 'timestamps.npy'
        path = os.path.join(
            "..", "data", "experiments", self.experiment_name, "extracted_data", file_name
        )
        try:
            old_timestamps = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            old_timestamps = np.array([])
        new_timestamp = np.array([node_name, message, get_current_timestamp(), time.time()])

        np.save(path, np.append(old_timestamps, new_timestamp))

    def save_state(self):
        """Save the state of the current stage."""
        self.save_timestamp(message='save_state')

        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        # file_name = self.name + "_" + current_name + ".pkl"
        dir_structure = os.path.join("..", "data", "experiments", self.experiment_name, "node_states",
                                     *current_name.split('-'))
        os.makedirs(dir_structure, exist_ok=True)  # create directory structure if it doesn't already exist

        file_name = self.name + ".pkl"
        path = os.path.join(dir_structure, file_name)

        # path = os.path.join(
        #     "..", "data", "experiments", self.experiment_name, "node_states", file_name
        # )
        print(f"saving to path {path}")

        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]
        with open(path, "wb") as f:
            pickle.dump(self.current_candidate, f)

    def load_state(self):
        """Load the state of the current stage."""
        self.save_timestamp(message='load_state')

        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        dir_structure = os.path.join("..", "data", "experiments", self.experiment_name, "node_states",
                                     *current_name.split('-'))
        os.makedirs(dir_structure, exist_ok=True)  # create directory structure if it doesn't already exist

        file_name = self.name + ".pkl"
        path = os.path.join(dir_structure, file_name)
        # file_name = self.name + "_" + current_name + ".pkl"
        # path = os.path.join(
        #     "../data/experiments/", self.experiment_name, "node_states", file_name
        # )
        print(f"loading path {path}")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                candidate = pickle.load(f)
                self.update_candidate(candidate)

    def build_candidates(self, candidates_info: List[dict]) -> List[Candidate]:
        """
        Build new candidates from given info.

        Args:
            candidates_info (List[dict]): List of candidate info.

        Returns:
            List[Candidate]: List of built candidates.
        """
        candidates = []
        if self.state.current_candidate_idx is None:
            current_received_name = ""  # None #Should be ''
        else:
            current_received_name = self.state.candidate_names[
                self.state.current_candidate_idx
            ]
        for ids_count, candidate_info in enumerate(candidates_info):
            candidate = Candidate(
                sender=self.name,
                receiver=self.child_stages[0].name,
                sender_candidate_id=ids_count,
                name=str(current_received_name)
                + "-"  # replace with '-' to enable folder structure later
                + str(self.name)
                + "_"
                + str(ids_count),
                info=candidate_info,
            )
            candidates.append(candidate)
            if self.show_tree:
                this_name = self.name + "_" + str(ids_count)
                this_identifier = candidate.name
                print(f'creating node with name {this_name}')
                try:
                    if self.state.current_candidate_idx is None:
                        # self.data_access_layer.tree.create_node(this_name, this_name)
                        self.data_access_layer.tree.create_node(this_name, this_identifier)
                    else:
                        parent = self.state.received_candidates[
                            self.state.current_candidate_idx
                        ].sender
                        # parent_id = self.state.received_candidates[
                        #     self.state.current_candidate_idx
                        # ].sender_candidate_id
                        # parent_name = parent + "_" + str(parent_id)
                        # self.data_access_layer.tree.create_node(
                        #     this_name, this_name, parent=parent_name
                        # )
                        parent_identifier = self.state.received_candidates[
                            self.state.current_candidate_idx
                        ].name
                        self.data_access_layer.tree.create_node(
                            this_name, this_identifier, parent=parent_identifier
                        )
                except DuplicatedNodeIdError:
                    print('tree seems to be already built, moving on')
        if self.show_tree:
            self.data_access_layer.save_tree()
        return candidates


    def print_state(self):
        """Print the current state of the tuning stage."""

        print(f"current candidate idx: {self.state.current_candidate_idx}")
        print(
            f"current candidate name: {self.state.candidate_names[self.state.current_candidate_idx]}"
        )
        print(
            f"current candidate: {self.current_candidate}"
        )
        if self.show_tree:
            self.data_access_layer.tree.show()

    def investigate(self, candidate: Candidate):
        """Investigate a given candidate. To be implemented in subclasses."""

        pass

    def perform_measurements(self):
        """Perform measurements for the current stage. To be implemented in subclasses."""
        pass
    def on_measurement_start(self):
        self.current_candidate.times['measurement_start'] = [get_current_timestamp(), time.time()]
        message = f'Starting measurement for {self.name}, candidate name: {self.current_candidate.name}'
        self.data_access_layer.create_or_update_file(message)

    def on_measurement_end(self):
        self.current_candidate.times['measurement_end'] = [get_current_timestamp(), time.time()]
        self.current_candidate.data_taken = True
        self.save_state()

    def on_analysis_start(self):
        self.current_candidate.times['analysis_start'] = [get_current_timestamp(), time.time()]
        message = f'Starting analysis for {self.name}, candidate name: {self.current_candidate.name}'
        self.data_access_layer.create_or_update_file(message)

    def on_analysis_end(self, candidate_info: List[Dict]) -> List[Candidate]:
        self.current_candidate.times['analysis_end'] = [get_current_timestamp(), time.time()]
        candidates = self.build_candidates(candidate_info)
        self.current_candidate.resulting_candidates = candidates
        self.current_candidate.analysis_done = True
        self.save_state()

        return candidates

    def determine_candidates(self):
        """Determine candidates for the current stage. To be implemented in subclasses."""
        pass

    def receive_report(self, report):
        """Receive a report. To be implemented in subclasses."""
        pass


class TerminalStage(BaseStage):
    """Node that does not have any child nodes"""

    def __init__(
        self, experiment_name: str, configs: dict, data_access_layer: DataAccess
    ):
        """
        Initialise parameters that all stages have in common.

        Args:
            experiment_name (str): Name of the experiment.
            configs (dict): Configuration parameters for the tuning stage.
            data_access_layer (DataAccess): An instance to access data.
        """
        name = "Terminal"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.child_stages = None

    def investigate(self, candidate: Candidate):
        print("Terminal node reached, no further investigation needed")
        pass


class RootStage(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.left_plunger_voltage = self.configs["left_plunger_voltage"]
        self.right_plunger_voltage = self.configs["right_plunger_voltage"]

        # self.VSD = qcodes_parameters["VSD"]
        # self.VL = qcodes_parameters['VL']
        # self.VM = qcodes_parameters['VM']
        # self.VR = qcodes_parameters['VR']

        # candidates = self.build_candidates([{'left_plunger_voltage': left_plunger_voltage,
        #                                      'right_plunger_voltage': right_plunger_voltage}
        #                                     ])
        #
        # self.state.candidates = candidates

        self.parent_nodes = None

    def kick_off(self):
        left_barrier = self.configs['left_barrier_voltage']
        right_barrier = self.configs['right_barrier_voltage']
        middle_barrier = self.configs['middle_barrier_voltage']

        self.station['V_L'](left_barrier)
        self.station['V_M'](middle_barrier)
        self.station['V_R'](right_barrier)

        saved_name =copy(self.name)
        self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        self.name=saved_name
        self.add_candidate(root_candidate)

        if self.configs['bias_voltage']>0:
            candidates = self.build_candidates(
                [
                    {
                        "left_plunger_voltage": self.left_plunger_voltage,
                        "right_plunger_voltage": self.right_plunger_voltage,
                        "bias_direction": "positive_bias",
                    }
                ]
            )
        else:
            candidates = self.build_candidates(
                [
                    {
                        "left_plunger_voltage": self.left_plunger_voltage,
                        "right_plunger_voltage": self.right_plunger_voltage,
                        "bias_direction": "negative_bias",
                    },
                ]
            )
        # candidates = self.build_candidates(
        #     [
        #         {
        #             "left_plunger_voltage": self.left_plunger_voltage,
        #             "right_plunger_voltage": self.right_plunger_voltage,
        #             "bias_direction": "positive_bias",
        #         },
        #         {
        #             "left_plunger_voltage": self.left_plunger_voltage,
        #             "right_plunger_voltage": self.right_plunger_voltage,
        #             "bias_direction": "negative_bias",
        #         },
        #     ]
        # )
        # root_candidate = self.build_candidates([{}])[0]

        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate()

    def investigate(self):
        candidates = self.state.received_candidates[0].resulting_candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            self.station['V_SD'](self.configs["bias_voltage"])
            # if candidate.info["bias_direction"] == "positive_bias":
            #     self.station['V_SD'](self.configs["bias_voltage"])
            # elif candidate.info["bias_direction"] == "negative_bias":
            #     self.station['V_SD'](-self.configs["bias_voltage"])
            # else:
            #     raise NotImplementedError
            self.child_stages[0].investigate(candidate)



class RootStageFromScratch(BaseStage):
    def __init__(
        self,
        experiment_name: str,
        qcodes_parameters,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True

        self.parent_nodes = None

    def kick_off(self):
        saved_name =copy(self.name)
        self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        self.name=saved_name
        self.add_candidate(root_candidate)

        candidates = self.build_candidates([{'bias_direction': 'positive_bias'}])

        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate(candidates)

    def investigate(self, candidates):
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)


class RootStageFromBarrierVolume(BaseStage):
    def __init__(
        self,
            station,
        experiment_name: str,
        qcodes_parameters,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.parent_nodes = None

    def is_inside_convex_hull(self, points, new_point):
        # print(len(points), new_point)
        original_hull = ConvexHull(points)
        original_volume = original_hull.volume

        # Add the new point
        updated_points = list(points) + [new_point]
        updated_hull = ConvexHull(updated_points)
        updated_volume = updated_hull.volume
        # print(original_volume, updated_volume)
        return original_volume == updated_volume

    def kick_off(self):
        saved_name =copy(self.name)
        self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        self.name=saved_name
        self.add_candidate(root_candidate)

        barrier_locs_list = self.configs["barrier_locs_list"]
        major_axis_poff = self.configs["major_axis_poff"]
        major_axis_poff_new_set = self.configs["major_axis_poff_new_set"]

        barrier_locs_normalised = np.array(barrier_locs_list) / major_axis_poff

        barrier_locs_rescaled = barrier_locs_normalised * major_axis_poff_new_set

        lower_bounds = [np.min(barrier_locs_rescaled[:, idx_b]) for idx_b in range(3)]
        upper_bounds = [np.max(barrier_locs_rescaled[:, idx_b]) for idx_b in range(3)]

        np.random.seed(1)
        sampler = qmc.Sobol(d=3, scramble=False)
        sample = sampler.random_base2(m=6)  # 7)
        sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

        # random_points *= 0.2
        print('random points generated')
        inside = []
        outside = []
        for point in sample_scaled:
            in_vol = self.is_inside_convex_hull(barrier_locs_rescaled, point)
            if in_vol:
                inside.append(point)
            else:
                outside.append(point)
        outside = np.array(outside)
        inside = np.array(inside)
        print(f'found {len(outside)} outside the hull and {len(inside)} inside')

        self.bias_voltage = self.configs["bias_voltage"]

        candidates_info = []
        for point in inside:
            for bias_direction in self.configs['bias_directions']:
                c_info = {'V_L': point[0],
                          'V_M': point[1],
                          'V_R': point[2],
                          'bias_direction': bias_direction
                          }
                candidates_info.append(c_info)

        candidates = self.build_candidates(candidates_info )

        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate(candidates)

    def investigate(self, candidates):
        for candidate in candidates:
            for barrier_name in ['V_L', 'V_M', 'V_R']:
                self.station[barrier_name](candidate.info[barrier_name])
            sign = 1 if candidate.info['bias_direction']=='positive_bias' else -1
            self.station['V_SD'](self.bias_voltage * sign)

            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

class RootStageFromROI(BaseStage):
    def __init__(
        self,
        experiment_name: str,
        qcodes_parameters,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True

        self.parent_nodes = None

    def kick_off(self):
        saved_name =copy(self.name)
        self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        self.name=saved_name
        self.add_candidate(root_candidate)

        candidates = self.build_candidates([{
            "lower_bound": self.configs['lower_bound'],
            "upper_bound": self.configs['upper_bound'],
            'bias_direction': 'positive_bias'
        }]
        )

        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate(candidates)

    def investigate(self, candidates):
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)



class RootStageForQubitMapping(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.parent_nodes = None

    def kick_off(self, investigation_result):
        left_barrier = self.configs['left_barrier_voltage']
        right_barrier = self.configs['right_barrier_voltage']
        middle_barrier = self.configs['middle_barrier_voltage']

        self.station['V_L'](left_barrier)
        self.station['V_M'](middle_barrier)
        self.station['V_R'](right_barrier)

        self.station['V_SD'](self.configs['bias_voltage'])

        saved_name = copy(self.name)
        self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        self.name = saved_name
        self.add_candidate(root_candidate)
        left_plunger_voltage = self.configs['left_plunger_voltage']
        right_plunger_voltage = self.configs['right_plunger_voltage']
        candidates = self.build_candidates(
            [
                {
                    "voltage_location": {
                        "left_plunger_voltage": left_plunger_voltage,
                        "right_plunger_voltage": right_plunger_voltage,
                    },
                    "bias_direction": "positive_bias",
                    "freq_vs": 2.79e9,
                    "burst_time": 5e-9,
                    "magnetic_field": 0.3,
                }])


        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate(candidates, investigation_result)

    def investigate(self, candidates, investigation_result):
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            investigation_result = self.child_stages[0].investigate(candidate, investigation_result)
        return investigation_result

class RootStageForCurrentOptimisation(BaseStage):
    def __init__(
            self,
            station,
            experiment_name: str,
            qcodes_parameters,
            configs: dict,
            data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.parent_nodes = None

    def run_investigation(self, i, candidates_info):
        print(f'i {i}, {self.state}')
        investigation_result = InvestigationResult()
        left_barrier = candidates_info['barrier_voltages']['V_L']
        right_barrier = candidates_info['barrier_voltages']['V_R']
        middle_barrier = candidates_info['barrier_voltages']['V_M']

        self.station['V_L'](left_barrier)
        self.station['V_M'](middle_barrier)
        self.station['V_R'](right_barrier)

        self.station['V_SD'](self.configs['bias_voltage'])

        # saved_name = copy(self.name)
        # self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        # self.name = saved_name
        self.add_candidate(root_candidate)
        # self.add_candidate(candidate)

        left_plunger_voltage = candidates_info['centroid_triangle']['left_plunger_voltage']
        right_plunger_voltage = candidates_info['centroid_triangle']['right_plunger_voltage']
        bias_direction = 'positive_bias' if self.configs['bias_voltage']>0 else 'negative_bias'
        candidates = self.build_candidates([
            {
                "barrier_voltages": candidates_info['barrier_voltages'],
                "voltage_location": {
                    "left_plunger_voltage": left_plunger_voltage,
                    "right_plunger_voltage": right_plunger_voltage,
                },
                "bias_direction": bias_direction,
            }])
        # self.name = saved_name
        # self.add_candidate(candidate)
        # self.load_state()
        self.current_candidate.resulting_candidates.append(candidates[0])
        # self.state.received_candidates[0].resulting_candidates = candidates
        self.print_state()
        investigation_result = self.child_stages[0].investigate(candidates[0], investigation_result)
        self.save_state()
        return investigation_result

class RootStageForDetuningGathering(BaseStage):
    def __init__(
            self,
            station,
            experiment_name: str,
            qcodes_parameters,
            configs: dict,
            data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.parent_nodes = None

    def run_investigation(self, i, candidates_info):
        print(f'i {i}, {self.state}')
        investigation_result = InvestigationResult()
        left_barrier = candidates_info['barrier_voltages']['V_L']
        right_barrier = candidates_info['barrier_voltages']['V_R']
        middle_barrier = candidates_info['barrier_voltages']['V_M']

        self.station['V_L'](left_barrier)
        self.station['V_M'](middle_barrier)
        self.station['V_R'](right_barrier)

        self.station['V_SD'](self.configs['bias_voltage'])

        # saved_name = copy(self.name)
        # self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        # self.name = saved_name
        self.add_candidate(root_candidate)
        # self.add_candidate(candidate)

        left_plunger_voltage = candidates_info['centroid_triangle']['left_plunger_voltage']
        right_plunger_voltage = candidates_info['centroid_triangle']['right_plunger_voltage']
        candidates = self.build_candidates([
            {
                "barrier_voltages": candidates_info['barrier_voltages'],
                "voltage_location": {
                    "left_plunger_voltage": left_plunger_voltage,
                    "right_plunger_voltage": right_plunger_voltage,
                },
                "bias_direction": "positive_bias",
                "freq_vs": 2.79e9,
                "burst_time": candidates_info['burst_time'],
                "magnetic_field": candidates_info['magnetic_field'],
                'guid_high_res': candidates_info['guid_high_res']
            }])
        # self.name = saved_name
        # self.add_candidate(candidate)
        # self.load_state()
        self.current_candidate.resulting_candidates.append(candidates[0])
        # self.state.received_candidates[0].resulting_candidates = candidates
        self.print_state()
        investigation_result = self.child_stages[0].investigate(candidates[0], investigation_result)
        self.save_state()
        return investigation_result
class RootStageForQubitMapInvestigation(BaseStage):
    def __init__(
            self,
            station,
            experiment_name: str,
            qcodes_parameters,
            configs: dict,
            data_access_layer: DataAccess,
    ):
        name = "Root"  # "RootSuggestor"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True
        self.station = station
        self.parent_nodes = None

    def run_investigation(self, i, candidates_info):
        print(f'i {i}, {self.state}')
        investigation_result = InvestigationResult()
        left_barrier = candidates_info['barrier_voltages']['V_L']
        right_barrier = candidates_info['barrier_voltages']['V_R']
        middle_barrier = candidates_info['barrier_voltages']['V_M']

        self.station['V_L'](left_barrier)
        self.station['V_M'](middle_barrier)
        self.station['V_R'](right_barrier)

        self.station['V_SD'](self.configs['bias_voltage'])

        # saved_name = copy(self.name)
        # self.name = 'root'
        root_candidate = self.build_candidates([{}])[0]
        # self.name = saved_name
        self.add_candidate(root_candidate)
        # self.add_candidate(candidate)

        left_plunger_voltage = candidates_info['centroid_triangle']['left_plunger_voltage']
        right_plunger_voltage = candidates_info['centroid_triangle']['right_plunger_voltage']
        candidates = self.build_candidates([
            {
                "barrier_voltages": candidates_info['barrier_voltages'],
                "voltage_location": {
                    "left_plunger_voltage": left_plunger_voltage,
                    "right_plunger_voltage": right_plunger_voltage,
                },
                "bias_direction": self.configs['bias_direction'],
                "freq_vs": 2.79e9,
                "burst_time": candidates_info['burst_time'],
                "magnetic_field": candidates_info['magnetic_field'],
            }])
        # self.name = saved_name
        # self.add_candidate(candidate)
        # self.load_state()
        self.current_candidate.resulting_candidates.append(candidates[0])
        # self.state.received_candidates[0].resulting_candidates = candidates
        self.print_state()
        investigation_result = self.child_stages[0].investigate(candidates[0], investigation_result)
        self.save_state()
        return investigation_result

    # def build_candidates(self, candidate_info: List[dict]) -> List[Candidate]:
    #     """
    #     Build new candidates from given info.
    #
    #     Args:
    #         candidates_info (List[dict]): List of candidate info.
    #
    #     Returns:
    #         List[Candidate]: List of built candidates.
    #     """
    #     current_received_name = ""  # None #Should be ''
    #     if self.current_candidate is not None:
    #         this_idx = len(self.current_candidate.resulting_candidates)
    #     else:
    #         this_idx = 0
    #     candidate = Candidate(
    #         sender=self.name,
    #         receiver=self.child_stages[0].name,
    #         sender_candidate_id=this_idx,
    #         name=str(current_received_name)
    #              + "-"  # replace with '-' to enable folder structure later
    #              + str(self.name)
    #              + "_"
    #              + str(this_idx),
    #         info=candidate_info,
    #     )
    #
    #     return candidate

