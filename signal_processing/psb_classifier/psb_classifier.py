import torchvision
from torch import nn
import torch
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        """
        Initialize the LeNet5 model for image classification. Assumes 100x100 pixel images.

        Args:
            num_classes (int, optional): The number of output classes. Defaults to 2.
        """
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7744, 120)  # assumes input of 100x100 images
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LeNet5 model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def get_net(
    device: torch.device,
    model_type: str = "resnet18",
) -> nn.Module:
    """
    Gets the network, optimizer, scheduler, and criterion based on the specified parameters.

    Args:
        device (torch.device): The device to which the model should be sent.
        class_weights (torch.Tensor, optional): A tensor of class weights. If None, all classes are assumed to have equal weight.
        model_type (str, optional): The type of the model to use. Can be 'resnet18' or 'lenet'. Defaults to 'resnet18'.

    Returns:
        Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau, nn.Module]: The model, optimizer, scheduler, and criterion.
    """
    if model_type == "resnet18":
        net = torchvision.models.resnet18(pretrained=False)
        # net = torchvision.models.resnet34(pretrained=False)
        net.conv1 = nn.Conv2d(
            2,
            net.conv1.out_channels,
            net.conv1.kernel_size,
            net.conv1.stride,
            net.conv1.padding,
            bias=net.conv1.bias,
        )

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
    elif model_type == "lenet":
        net = LeNet5()
    else:
        raise (f"model_type {model_type} not considered")
    net = net.to(device)
    return net




class EnsembleClassifier:
    def __init__(
        self,
        folder_path_to_nn: str,
        ensemble_size: int = 10,
        majority_vote: bool = False,
        model_type:str = 'lenet',
        network_names:str = 'only_simulator',
        verbose:bool=False
    ):
        """Create an ensemble PSB classifier.

        Go through the demo notebook for more info on how to use it.

        Args:
            folder_path_to_nn: Points to the location of the stored neural networks
            ensemble_size: Number of individual classifiers
            majority_vote: Decides how outputs of individual classifiers are
                           combined. In a majority vote, each classifier gets one
                           vote. If this is set to false, the raw output of all classifiers
                           (number between 0 and 1) will be averaged and returned.
        """
        self.majority_vote = majority_vote
        self.device = torch.device("cpu")
        self.model_type = model_type
        self.verbose = verbose
        self.ensemble = []
        for rep in range(ensemble_size):
            path_to_nn = folder_path_to_nn + network_names + "_rep_" + str(rep) + ".pth"
            # net = torchvision.models.resnet18(pretrained=False)
            # net.conv1 = nn.Conv2d(
            #     2,
            #     net.conv1.out_channels,
            #     net.conv1.kernel_size,
            #     net.conv1.stride,
            #     net.conv1.padding,
            #     bias=net.conv1.bias,
            # )
            #
            # num_ftrs = net.fc.in_features
            # net.fc = nn.Linear(num_ftrs, 2)
            # net = net.to(self.device)
            net = get_net(self.device, model_type=model_type)
            net.load_state_dict(
                torch.load(path_to_nn, map_location=torch.device("cpu"))
            )
            net.eval()
            self.ensemble.append(net)
        return

    def __call__(self, X):
        """Predict PSB in sample X.

        warning, normalisation assumes only one sample is passed at a time.
        Args:
            X: Sample of PSB to be predicted. Expected shape:
               (number_of_samples, 2 [blocked/unblocked], 100, 100)
        """
        X = self.normalise(X)



        with torch.no_grad():
            pred_array = []
            for net in (self.ensemble):
                outputs = net(torch.Tensor(X).to(self.device))
                if self.majority_vote:
                    pred_array.append(
                        torch.max(outputs.data, 1).indices.detach().cpu().numpy()
                    )
                else:
                    activation = nn.Softmax(dim=1)
                    pred_array.append(activation(outputs)[:, -1].detach().cpu().numpy())
                if self.verbose:
                    print(f'pred_array[-1]{pred_array[-1]}')
        return np.mean(pred_array, axis=0)

    def predict(self, X):

        return self(X)

    def normalise(self, all_imgs, maximum=None, minimum=None, joint_norm = False):
        normed_imgs = []



        for img in all_imgs:
            img = np.asarray(img)
            if joint_norm:
                if maximum == None:
                    maximum = np.max(img)
                if minimum == None:
                    minimum = np.min(img)
                normed_imgs.append((img - minimum) / (maximum - minimum))
            else:
                img0 = img[0]
                img0 = (img0 - np.min(img0)) / (np.max(img0) - np.min(img0))
                img1 = img[1]
                img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
                normed_imgs.append([img0, img1])
        return np.array(normed_imgs)

    def cutout(self, img: np.ndarray, blob: np.ndarray, sidelength: int = 50) -> np.ndarray:
        """
        Cuts out a square region from the image centered around the blob.

        Args:
            img (np.ndarray): The input image.
            blob (np.ndarray): The blob around which to cut out.
            sidelength (int, optional): The sidelength of the square to cut out. Defaults to 50.

        Returns:
            np.ndarray: The cut out image.
        """
        x_bottom = (
            int(blob[0] - sidelength / 2) if int(blob[0] - sidelength / 2) >= 0 else 0
        )
        x_top = int(blob[0] + sidelength / 2) if int(blob[0] + sidelength / 2) >= 0 else 0
        y_bottom = (
            int(blob[1] - sidelength / 2) if int(blob[1] - sidelength / 2) >= 0 else 0
        )
        y_top = int(blob[1] + sidelength / 2) if int(blob[1] + sidelength / 2) >= 0 else 0
        img_save = img.copy()
        img = img[x_bottom:x_top, y_bottom:y_top]
        while img.shape[0] != img.shape[1]:
            sidelength -= 1
            img = self.cutout(img_save, blob, sidelength)
        return img


    def predict_from_large_scan(self, img_block, img_leak, locations, sidelength=20, flip_bias = False):
        predictions = []

        for location in tqdm(locations):
            img_block_cutout = self.cutout(img_block, location, sidelength= sidelength)
            img_leak_cutout = self.cutout(img_leak, location, sidelength= sidelength)
            if flip_bias:
                img_block_cutout = img_block_cutout.T
                img_leak_cutout = img_leak_cutout.T
            if img_block_cutout.shape[0] == sidelength and img_block_cutout.shape[1] == sidelength:
                el0 = resize(img_block_cutout, (100,100))
                el1 = resize(img_leak_cutout, (100,100))
                prediction = self.predict([[el0, el1]])
                predictions.append(prediction)
            else:
                predictions.append(None)
            if self.verbose:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2,1)
                axs[0].imshow(img_block_cutout, origin = 'lower')
                axs[0].set_title('block')
                axs[1].imshow(img_leak_cutout, origin = 'lower')
                axs[1].set_title('leaky')
                plt.suptitle(f'prediction {predictions[-1]}')
                plt.show()


        return predictions


    def predict_from_large_scan_doesntwork(self, img_block, img_leak, locations, sidelength=20, flip_bias = False):

        imgs = []
        ignore = []
        for location in locations:
            if self.verbose:
                print(f'location {location}')
            img_block_cutout = self.cutout(img_block, location, sidelength= sidelength)
            img_leak_cutout = self.cutout(img_leak, location, sidelength= sidelength)
            if flip_bias:
                img_block_cutout = img_block_cutout.T
                img_leak_cutout = img_leak_cutout.T
            # if self.verbose:
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots(2,1)
            #     axs[0].imshow(img_block_cutout, origin = 'lower')
            #     axs[0].set_title('block')
            #     axs[1].imshow(img_leak_cutout, origin = 'lower')
            #     axs[1].set_title('leaky')
            #     plt.show()

            if img_block_cutout.shape[0] == sidelength and img_block_cutout.shape[1] == sidelength:
                ignore.append(False)
            else:
                img_block_cutout = np.zeros((100,100))
                img_leak_cutout = np.ones((100,100))
                ignore.append(True)
            if self.verbose:
                print(f'img_block_cutout.shape {img_block_cutout.shape}')
                print(f'img_leak_cutout.shape {img_leak_cutout.shape}')

            el0 = resize(img_block_cutout, (100,100))
            el1 = resize(img_leak_cutout, (100,100))

            imgs.append([el0, el1])

        if self.verbose:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 10, figsize=(10,3))
            for i, x in enumerate(imgs):
                axs[0][(i+1)%10].imshow(x[0], origin='lower')
                axs[0][(i+1)%10].set_title('block')
                axs[1][(i+1)%10].imshow(x[1], origin='lower')
                axs[1][(i+1)%10].set_title('leaky')
                if i ==9:
                    break
            plt.show()

        predictions = self.predict(imgs)
        if self.verbose:
            print(f'predictions. {predictions}')
            print(f'ignore. {ignore}')
        for i, (pred, ign) in enumerate(zip(predictions, ignore)):
            if ign:
                predictions[i] = None

        # predictions = []
        # for img in tqdm(imgs):
        #     [img_block_cutout, img_leak_cutout] = img
        #     if img_block_cutout.shape[0] == sidelength and img_block_cutout.shape[1] == sidelength:
        #         el0 = resize(img_block_cutout, (100,100))
        #         el1 = resize(img_leak_cutout, (100,100))
        #         prediction = self.predict([[el0, el1]])
        #         # predictions.append(prediction)
        #     else:
        #         predictions.append(None)
        return predictions


if __name__=='__main__':
    import os
    import qcodes as qc
    import toml
    from qcodes import (Measurement,
                        experiments,
                        initialise_database,
                        initialise_or_create_database_at,
                        load_by_guid,
                        load_by_run_spec,
                        load_experiment,
                        load_last_experiment,
                        load_or_create_experiment,
                        new_experiment,
                        ManualParameter)

    import sys

    sys.path.append('../../pipelines')
    import matplotlib.pyplot as plt

    from utils import get_current_timestamp, timestamp_files, read_row_from_file, draw_boxes_and_preds

    experiment_name = '20230530_live_test'
    comment = 'first attempt at search for spin physics'
    config_path = '../../data/config_files/v1.toml'

    # save
    configs = toml.load(config_path)

    configs_loc_det = configs['location_detection']

    db_name = "../../pipelines/GeSiNW_Qubit_VTI01_Jonas.db"  # Database name
    sample_name = "Butch"  # Sample name
    exp_name = "Qubit_Search"  # Experiment name

    db_file_path = os.path.join(os.getcwd(), db_name)
    qc.config.core.db_location = db_file_path
    initialise_or_create_database_at(db_file_path)

    experiment = load_or_create_experiment(experiment_name=exp_name,
                                           sample_name=sample_name)



    from qcodes import load_by_run_spec

    # data_I_SD = load_by_run_spec(captured_run_id=5).to_xarray_dataset()['I_SD']
    # data_I_SD_low_magnet = load_by_run_spec(captured_run_id=6).to_xarray_dataset()['I_SD']

    data_I_SD = load_by_run_spec(captured_run_id=7).to_xarray_dataset()['I_SD']
    data_I_SD_low_magnet = load_by_run_spec(captured_run_id=8).to_xarray_dataset()['I_SD']

    msmt_id = 5
    # %%
    window_rp = configs_loc_det['window_right_plunger']
    window_lp = configs_loc_det['window_left_plunger']
    resolution = configs_loc_det['resolution']

    lp = configs_loc_det['left_plunger_voltage']
    rp = configs_loc_det['right_plunger_voltage']

    rp_start = rp - window_rp / 2
    rp_end = rp + window_rp / 2

    lp_start = lp - window_lp / 2
    lp_end = lp + window_lp / 2

    n_px_rp = int(window_rp / resolution)
    n_px_lp = int(window_lp / resolution)

    x_array = np.linspace(lp_start, lp_end, n_px_lp)
    y_array = np.linspace(rp_start, rp_end, n_px_rp)
    # %%
    # detect bias triangle locations
    from signal_processing.bias_triangle_processing.bias_triangle_detection import btriangle_location_detection

    # anchor, peaks_px, peaks, all_triangles_px, all_triangles, fig = btriangle_location_detection.get_locations(
    #     -data_I_SD.to_numpy().T, x_array, y_array, 'LP', 'RP', return_figure=True, plot=True)
    anchor, peaks_px, peaks, all_triangles_px, all_triangles, fig = btriangle_location_detection.get_locations(
        data_I_SD.to_numpy().T, x_array, y_array, 'LP', 'RP', return_figure=True, plot=True)

    psb_classifier = EnsembleClassifier('saved_networks/lenet_only_sim/', verbose = False)
    # psb_classifier = EnsembleClassifier('saved_networks/lenet_all_mixed_data_finfets/', network_names = 'all_mixed_data')

    # psb_classifier = EnsembleClassifier('saved_networks/', verbose = True, model_type = 'resnet18',
    #                                     ensemble_size = 1)

    # values = -data_I_SD.to_numpy().T
    # values_low_magnet = -data_I_SD_low_magnet.to_numpy().T
    values = data_I_SD.to_numpy().T
    values_low_magnet = data_I_SD_low_magnet.to_numpy().T

    sidelength = np.max(peaks_px)
    predictions = psb_classifier.predict_from_large_scan(values_low_magnet,
                                                         values,
                                                         all_triangles_px,
                                                         sidelength=sidelength, flip_bias = True)

    print(predictions)


    fig = draw_boxes_and_preds(values, values_low_magnet, all_triangles_px, predictions, sidelength)
    plt.show()