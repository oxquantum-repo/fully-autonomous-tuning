import os
import pickle
import time
from copy import copy
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Union

import numpy as np

from helper_functions.data_access_layer import DataAccess
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
    data_identifiers: Dict = field(
        default_factory=dict
    )  # store link to previously taken data here, either a GUID or a name that can be loaded via DataAccess
    resulting_candidates: List = field(default_factory=list)

    def __str__(self):
        """For prettier prints"""
        return (
            f"Candidate(\n"
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
            f")"
        )


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

    def save_timestamp(self, message: str = ""):
        """Save the state of the current stage."""
        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        node_name = self.name + "_" + current_name + ".txt"
        file_name = "timestamps.npy"
        path = os.path.join(
            self.data_access_layer.root_folder,
            self.experiment_name,
            "extracted_data",
            file_name,
        )
        try:
            old_timestamps = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            old_timestamps = np.array([])
        new_timestamp = np.array(
            [node_name, message, get_current_timestamp(), time.time()]
        )

        np.save(path, np.append(old_timestamps, new_timestamp))

    def save_state(self):
        """Save the state of the current stage."""
        self.save_timestamp(message="save_state")

        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        dir_structure = os.path.join(
            self.data_access_layer.root_folder,
            self.experiment_name,
            "node_states",
            *current_name.split("-"),
        )
        os.makedirs(
            dir_structure, exist_ok=True
        )  # create directory structure if it doesn't already exist

        file_name = self.name + ".pkl"
        path = os.path.join(dir_structure, file_name)

        print(f"saving to path {path}")

        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]
        with open(path, "wb") as f:
            pickle.dump(self.current_candidate, f)

    def load_state(self):
        """Load the state of the current stage."""
        self.save_timestamp(message="load_state")

        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        dir_structure = os.path.join(
            self.data_access_layer.root_folder,
            self.experiment_name,
            "node_states",
            *current_name.split("-"),
        )
        os.makedirs(
            dir_structure, exist_ok=True
        )  # create directory structure if it doesn't already exist

        file_name = self.name + ".pkl"
        path = os.path.join(dir_structure, file_name)

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

        return candidates

    def print_state(self):
        """Print the current state of the tuning stage."""

        print(f"current candidate idx: {self.state.current_candidate_idx}")
        print(
            f"current candidate name: {self.state.candidate_names[self.state.current_candidate_idx]}"
        )
        print(f"current candidate: {self.current_candidate}")

    def investigate(self, candidate: Candidate):
        """Investigate a given candidate. To be implemented in subclasses."""
        pass

    def investigate(self, candidate: Candidate):
        """Investigate a given candidate.

        1. Add candidate to the list of received candidates.
        2. Check if data has been taken for the candidate.
        3. Perform measurements if data has not been taken.
        4. Determine new candidates based on the measurements.
        5. Send candidates to child nodes.

        Overwrite in subclasses if more complicated logic is needed."""
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        self.current_candidate.resulting_candidates = candidates

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self):
        """Perform measurements for the current stage. To be implemented in subclasses."""
        pass

    def on_measurement_start(self):
        self.current_candidate.times["measurement_start"] = [
            get_current_timestamp(),
            time.time(),
        ]

    def on_measurement_end(self):
        self.current_candidate.times["measurement_end"] = [
            get_current_timestamp(),
            time.time(),
        ]
        self.current_candidate.data_taken = True
        self.save_state()

    def on_analysis_start(self):
        self.current_candidate.times["analysis_start"] = [
            get_current_timestamp(),
            time.time(),
        ]

    def on_analysis_end(self, candidate_info: List[Dict]) -> List[Candidate]:
        self.current_candidate.times["analysis_end"] = [
            get_current_timestamp(),
            time.time(),
        ]
        candidates = self.build_candidates(candidate_info)
        self.current_candidate.resulting_candidates = candidates
        self.current_candidate.analysis_done = True
        self.save_state()

        return candidates

    def determine_candidates(self):
        """Determine candidates for the current stage. To be implemented in subclasses."""
        pass



class RootStage(BaseStage):
    def __init__(
        self,
        experiment_name: str,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Root"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.state.data_taken = True

        self.parent_nodes = None

    def kick_off(self):
        saved_name = copy(self.name)
        self.name = "root"
        root_candidate = self.build_candidates([{}])[0]
        self.name = saved_name
        self.add_candidate(root_candidate)
        candidates = self.build_candidates(
            [
                {"info": "info"}
            ]
        )
        self.state.received_candidates[0].resulting_candidates = candidates

        self.investigate()

    def investigate(self):
        candidates = self.state.received_candidates[0].resulting_candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name}, sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)
