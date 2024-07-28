from typing import Dict, List

from mindfoundry.optaas.client.parameter import FloatParameter


def get_readout_optim_params(
    params_config: Dict,
):
    parameters = [
        FloatParameter(name, **param_config, id=name)
        for name, param_config in params_config.items()
    ]
    return parameters


def get_tangent_space_params(
    tangent_volt_window: float, norm_volt_window: float, num_dimensions: int = 3
) -> List[FloatParameter]:
    # Parameter we can tune
    parameters = [
        FloatParameter(
            f"tangent_param_{i}",
            minimum=-tangent_volt_window / 2,
            maximum=tangent_volt_window / 2,
            id=f"tangent_param_{i}",
        )
        for i in range(num_dimensions - 1)
    ]
    parameters.append(
        FloatParameter(
            "norm_param",
            minimum=-norm_volt_window / 2,
            maximum=norm_volt_window / 2,
            id="norm_param",
        )
    )
    return parameters
