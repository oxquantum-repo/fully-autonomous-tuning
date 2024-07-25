import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from bias_triangle_detection.bayesian_optimization.setup import setup
from loguru import logger
from mindfoundry.optaas.client.result import Configuration, Result
from mindfoundry.optaas.client.task import Task
from mindfoundry.optaas.client.user_defined_configuration import (
    UserDefinedConfiguration,
)
from tqdm import tqdm

try:
    from helper_functions.data_access_layer import DataAccess
    from pipelines.utils import timestamp_files
except:
    print(f"DataAccess and timestamp_files could not be loaded")

logger.add(sys.stderr, level="INFO", filter=__name__)


@dataclass(order=True)
class EvaluationResult:
    score: float
    config: Dict[str, float] = field(compare=False)
    metadata: Dict[str, float] = field(compare=False, default_factory=dict)


Scorer = Callable[[Dict[str, float]], EvaluationResult]


def get_next_configurations(
    task: Task,
    orig_configs: List[Configuration],
    batch_results: List[EvaluationResult],
) -> List[Configuration]:
    # if batch configs have not changed, return resutls
    if all(
        orig_config.values == result.config
        for orig_config, result in zip(orig_configs, batch_results)
    ):
        results = [
            Result(config, score=result.score, user_defined_data=result.metadata)
            for config, result in zip(orig_configs, batch_results)
        ]
        return task.record_results(results)
    user_configs = [
        UserDefinedConfiguration(
            result.config, score=result.score, user_defined_data=result.metadata
        )
        for result in batch_results
    ]
    task.add_user_defined_configuration(configurations=user_configs)
    return task.generate_configurations(len(user_configs))


@dataclass
class score_with_constraint_wrapper:
    constraint: Callable[[Dict[str, float]], bool]
    default_score: float = 0.0

    def __call__(self, func: Scorer) -> Scorer:
        def wrapper(**kwargs):
            if self.constraint(kwargs):
                return func(**kwargs)
            return EvaluationResult(score=self.default_score, config=kwargs)

        return wrapper


def get_scorer(
    point_to_score: Scorer,
    constraint: Optional[Callable[[Dict[str, float]], bool]] = None,
    default_score: float = -10,
) -> Scorer:
    if constraint is None:
        return point_to_score
    logger.warning(
        f"Setting default score to {default_score} for points outside given constraint"
    )
    return score_with_constraint_wrapper(constraint, default_score)(point_to_score)


def validate_score(scorer: Scorer, score_value: float):
    if not hasattr(scorer, "default_score"):
        return True
    return score_value > scorer.default_score


def run_optimization(
    optimizer: Task,
    scorer: Scorer,
    iterations: int = 30,
    batch_size: int = 1,
    log_every_n_steps: int = 3,
    max_total_iterations: int = None,
    break_on_cotunnelling: bool = False,
    data_access_layer: Optional["DataAccess"] = None,
) -> List[EvaluationResult]:
    max_total_iterations = max_total_iterations or iterations * 10
    results = []
    starting_msg = f'Running task "{optimizer.title}" for {iterations} iterations'
    logger.info(starting_msg)
    configurations = optimizer.generate_configurations(batch_size)
    best_config = None
    best_result = None
    n_valid_iterations = 0
    i = 0
    while n_valid_iterations < iterations and i < max_total_iterations:
        # for i in tqdm(range(iterations), desc="Volt space Optimization"):
        batch_results = []
        orig_configs = []
        cotunnelling_found = False
        for config in configurations:
            # Make the measurements there is one element at the time but they can be done in batches]
            evaluation_result = scorer(**config.values)
            batch_results.append(evaluation_result)
            results.append(evaluation_result)
            orig_configs.append(config)
            # Create the Result object
            score_value = evaluation_result.score
            if validate_score(scorer, score_value):
                n_valid_iterations += 1
            if break_on_cotunnelling and any(
                evaluation_result.metadata["is_axes_cotunneling"].values()
            ):
                cotunnelling_found = True
                break
            i += 1
        configurations = get_next_configurations(optimizer, orig_configs, batch_results)
        if (i % log_every_n_steps == 0) or (i == iterations - 1):
            best_result = optimizer.get_best_result_and_configuration()
            score_msg = f"Iteration: {i} | Best score: {best_result.score}"
            best_config = best_result.configuration.values
            logger.info(f"{score_msg} | Best Configuration: {best_config}")
            if data_access_layer is not None:
                data_access_layer.create_or_update_file(
                    f"{score_msg}\n best_result: {best_result}"
                )
                filename = timestamp_files() + "_optimisation_results"
                data_access_layer.save_data({filename: results})
                if data_access_layer.transform is not None: # True for readout optimisation
                    data_access_layer.plot_readout_scores(results)
        if cotunnelling_found:
            break
    return sorted(results, reverse=True)


if __name__ == "__main__":
    # optimize dummy func
    import numpy as np
    from bias_triangle_detection.bayesian_optimization.bayesian_optimizer import (
        run_optimization,
    )
    from bias_triangle_detection.bayesian_optimization.parameters import (
        get_tangent_space_params,
    )
    from bias_triangle_detection.bayesian_optimization.setup import setup

    tangent_volt_window = 0.05
    norm_volt_window = 0.1
    parameters = get_tangent_space_params(tangent_volt_window, norm_volt_window)
    client = setup()
    seeding = 5
    optimizer = client.create_task(
        title="Coulomb peak optim",
        parameters=parameters,  # parameters are independent of surface for now
        initial_configurations=seeding,
    )
    point_to_score = lambda **k: EvaluationResult(
        score=np.random.rand(),
        config={
            param.name: np.clip(np.random.rand(), param.minimum, param.maximum)
            for param in parameters
        },
        metadata={},
    )
    scorer = get_scorer(point_to_score, constraint=None, default_score=-20)

    run_optimization(
        optimizer,
        scorer,
    )
