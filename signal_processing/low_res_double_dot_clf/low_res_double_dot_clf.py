import numpy as np

# from .scoring.Last_score import final_score_cls
import time
import os

from torch import nn
import torch
import torchvision

from helper_functions.data_processing import normalise_array


# Initialise double_dot classifier
def initialise_low_res_classifier(
    configs: dict = None,
) -> tuple(torch.Tensor, torch.device):
    """Return low res current map nn classifier and pytorch device
    :param config: Experimental config file
    :type config: dict
    rtype: object, object
    """
    rel_dir = "double_dot_check_tools"

    default_model = "20220215_low_res_dqd_classifier_complete.pth"
    this_dir, _ = os.path.split(__file__)

    model_name = configs.get("low_res_classifier", default_model)
    model_path = os.path.join(this_dir, rel_dir, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torchvision.models.resnet18(pretrained=False)
    net.conv1 = nn.Conv2d(
        1,
        net.conv1.out_channels,
        net.conv1.kernel_size,
        net.conv1.stride,
        net.conv1.padding,
        bias=net.conv1.bias,
    )

    # Mount the weights onto ResNet
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    net.eval()
    print(model_path, " loaded")

    return net, device


def low_res_predict(low_res: np.ndarray, configs: dict = None) -> bool:

    low_res_norm = normalise_array(low_res)
    # Expand the dimensions to suit net.
    low_res_norm = low_res_norm[np.newaxis, np.newaxis, :]

    ddc, device = initialise_low_res_classifier(configs)
    outputs = ddc(torch.FloatTensor(low_res_norm).to(device))

    dd_found = torch.max(outputs.data, 1).indices.detach().cpu()

    return bool(dd_found)


# def low_res_and_score(data, minc, maxc, configs, **kwags):
#     fsc = final_score_cls(minc, maxc, configs["noise"], configs["segmentation_thresh"])

#     score = getattr(fsc, configs.get("mode", "score"))(
#         data, diff=configs.get("diff", 1)
#     )

#     s_cond = False

#     print("Score: %f" % score)

#     # Normalise Current map
#     # norm settings
#     low_res_norm = normalise_array(data)

#     # Expand the dimensions to suit net.
#     low_res_norm = low_res_norm[np.newaxis, np.newaxis, :]

#     # Brandon & Jonas Low Res Current map classifier
#     # Load Double Dot classifier
#     ddc, device = initialise_low_res_classifier(configs)

#     # Make a prediction if double dot is present
#     outputs = ddc(torch.FloatTensor(low_res_norm).to(device))
#     dd_found = torch.max(outputs.data, 1).indices.detach().cpu()
#     print("NN predicts a good double dot: ", bool(dd_found))

#     continue_investigation = bool(dd_found) or (score > 0.001)
#     return continue_investigation, bool(dd_found), None
