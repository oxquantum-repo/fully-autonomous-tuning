from typing import Dict, List, Tuple
from dataclasses import dataclass 
from loguru import logger

from mindfoundry.optaas.client.client import OPTaaSClient
from mindfoundry.optaas.client.parameter import FloatParameter
from mindfoundry.optaas.client.user_defined_configuration import (
    UserDefinedConfiguration,
)
from .bayesian_optimizer import EvaluationResult

class DimensionalityReducedError(Exception):
    pass

@dataclass 
class CoarseTuningOptimizer:
    """Manages the parameter bounds during coarse tuning and creates a new OPTaas optimizer with these.
    During this stage of calibration, we scan across the left and right plunge voltages at different points in a 3 dimensional 
    domain space for the left, right and middle bias voltages: (V_L,V_R,V_M).

    If cotunnelling is detected in during the left/right plunge scan, then the bounds of the domain 
    space are restricted to exclude that point. 
    
    For example, the initial bounds are (-3,3) for all dimensions of the domain space.
    A scan is performed at (X, Y, Z) and cotunelling is detected. The domain space bounds are 
    updated as follows:
        V_L -> (X, 3)
        V_R -> (Y, 3)
        V_M is unchanged. 

    This update is performed during initialization of this class from the EvalautionResults passed.
    Filtered results that are within the updated bounds are stored.
    """
    def __init__(self,results: List[EvaluationResult], initial_params: Dict[str, Tuple[float, float]], minimum_update: float, client: OPTaaSClient, optimizer_kwargs: Dict, update_params_ranges: bool = True):
        self.param_ranges = initial_params
        self.minimum_update = minimum_update
        if update_params_ranges:
            self._update_param_range_with_results(results)
        self.results = results
        self.filtered_results = self._filter_results_out_of_range(results) 
        self._print_info()        
        self.optimizer = client.create_task(
            title="Coulomb peak optim",
            parameters=self.optaas_parameters,
            **optimizer_kwargs,
        )
        self._pass_valid_results_to_optimizer()
        
    @property 
    def optaas_parameters(self) -> List[FloatParameter]:
        parameters = [
            FloatParameter(
                key,
                minimum=param_range[0],
                maximum=param_range[1],
                id=key
            )
            for key, param_range in self.param_ranges.items()
        ]
        return parameters
            

    def _update_param_range_with_results(self, results: List[EvaluationResult]):
        """Restricts the parameter range of each of the dimension by excluding any region that has cotunnelling"""
        for result in results:
            for param_name, is_cotunneling in result.metadata["is_axes_cotunneling"].items():
                if is_cotunneling:
                    current_min, current_max = self.param_ranges[param_name]
                    new_min = max(current_min, result.config[param_name] + self.minimum_update) 
                    if new_min > current_max:
                        raise DimensionalityReducedError(f"The presence of cotunneling exausted all the possible values for {param_name}")
                    self.param_ranges[param_name] = (new_min, current_max)
                    
                                  
    def _filter_results_out_of_range(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        filtered_results = []
        for result in results:
            if self.result_in_param_range(result):
                filtered_results.append(result)
        return filtered_results

    def _pass_valid_results_to_optimizer(self):
        if len(self.filtered_results) == 0:
            return              
        previous_run_results = [
            UserDefinedConfiguration(result.config, result.score, user_defined_data=result.metadata)
            for result in  self.filtered_results
        ]
        self.optimizer.add_user_defined_configuration(configurations=previous_run_results)
    
    def result_in_param_range(self, result: EvaluationResult):
        for param_key in self.param_ranges:
            if result.config[param_key] < self.param_ranges[param_key][0]:
                return False 
            if result.config[param_key] > self.param_ranges[param_key][1]:
                return False 
        return True

    def _print_info(self):
        logger.info(f"Coarse tuning optimizer constructed with parameters:")
        for key, param_range in self.param_ranges.items():
            logger.info(f"\t{key}: {param_range}")
        logger.info(f"Filtered {len(self.filtered_results)} results in range from previous optimizations.")
