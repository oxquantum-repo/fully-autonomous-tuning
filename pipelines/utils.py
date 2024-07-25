import logging
from datetime import datetime
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from qcodes_addons.Parameterhelp import CompensatedGateParameter
from qcodes_addons.AWGhelp import PulseParameter
from qcodes_addons.doNdAWG import do1dAWG, do2dAWG, init_Rabi

from helper_functions.pca import PCA


def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_files():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_compensated_gates(station):
    cpg_list = []
    for gate in [station['V_LP'], station['V_RP']]:
        if isinstance(gate, CompensatedGateParameter):
            cpg_list.append(gate)
    return cpg_list

def read_row_from_file(filename, row_num):
    with open(filename, "r") as f:
        lines = f.readlines()
    target_line = lines[row_num - 1]  # assuming row_num is 1-based
    values = target_line.strip().split()  # split the line into a list of values
    return np.array(values, dtype=float)  # convert the list into a numpy array


import seaborn
from skimage.draw import rectangle_perimeter

colormap = "icefire"

def plot_coarse_tuning_results(all_results, resulting_candidates=None, bounds=None, plot_2d=True, plot_max=False):
    if bounds is not None:
        lower_bound = bounds[0]
        upper_bound = bounds[1]
    if resulting_candidates is not None:
        v_points_candidates = {'V_L': [],
                               'V_R': [],
                               'V_M': []}
        for candidate in resulting_candidates:
            v_points_candidates['V_L'].append(candidate.info['config']['V_L'])
            v_points_candidates['V_M'].append(candidate.info['config']['V_M'])
            v_points_candidates['V_R'].append(candidate.info['config']['V_R'])
    v_points = {'V_L': [],
                'V_R': [],
                'V_M': []}
    scores = []

    cotunneling_detections = []
    for result in all_results:
        score = result.score
        scores.append(score)
        v_points['V_L'].append(result.config['V_L'])
        v_points['V_M'].append(result.config['V_M'])
        v_points['V_R'].append(result.config['V_R'])
        cotunneling_detections.append(result.metadata['is_axes_cotunneling'])
        cotunneling_detections[-1]['V_M'] = False
    max_score_idx = np.argmax(scores)
    print(len(cotunneling_detections))
    if plot_2d:
        fig = plt.figure(figsize=(6, 15))
        perspectives = [('V_L', 'V_M'), ('V_L', 'V_R'), ('V_R', 'V_M')]
        butch = {'V_L': 1.35,
                 'V_M': 0.66,
                 'V_R': 1.02, }
        if bounds is not None:
            lower_bound_dict = {'V_L': lower_bound[0],
                                'V_M': lower_bound[1],
                                'V_R': lower_bound[2], }
            upper_bound_dict = {'V_L': upper_bound[0],
                                'V_M': upper_bound[1],
                                'V_R': upper_bound[2], }
        for i, (x_name, y_name) in enumerate(perspectives):
            ax = fig.add_subplot(3, 1, i + 1)
            xdata = v_points[x_name]
            ydata = v_points[y_name]
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            # im = ax.scatter(xdata, ydata, c=np.log(scores), cmap="viridis")
            # im = ax.scatter(xdata, ydata, c=np.clip(scores, 0, 3), cmap="viridis")
            im = ax.scatter(xdata, ydata, c=scores, cmap="gnuplot")
            if resulting_candidates is not None:
                xdata_candidates = v_points_candidates[x_name]
                ydata_candidates = v_points_candidates[y_name]
                ax.scatter(xdata_candidates, ydata_candidates, c='none', edgecolor='white', s=100,
                       label='candidates for next stage')
            if plot_max:
                ax.scatter(xdata[max_score_idx], ydata[max_score_idx],s=100,c='white', marker='x', label='highest score')

            # butch_im = ax.scatter(butch[x_name], butch[y_name], marker='x', label='old qubit loc')
            if bounds is not None:
                ax.scatter(lower_bound_dict[x_name], lower_bound_dict[y_name], marker='*', label='lower bound')
                ax.scatter(upper_bound_dict[x_name], upper_bound_dict[y_name], marker='*', label='upper bound')

            # check each point in xdata and ydata
            for j in range(len(xdata)):
                if cotunneling_detections[j][x_name]:
                    ax.axvline(x=xdata[j], color='r', linestyle='--', label='cotunneling detected')
                if cotunneling_detections[j][y_name]:
                    ax.axhline(y=ydata[j], color='r', linestyle='--', label='cotunneling detected')

            plt.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        # plt.show()
    else:
        perspectives = [(30, 30), (30, 90), (30, 120), (30, 150)]
        fig = plt.figure(figsize=(4, 15))
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(4, 1, i + 1, projection="3d")
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            xdata = v_points['V_L']
            ydata = v_points['V_M']
            zdata = v_points['V_R']

            ax.set_xlabel("VL")
            ax.set_ylabel("VM")
            ax.set_zlabel("VR")
            im = ax.scatter3D(xdata, ydata, zdata, c=scores, cmap="viridis")
            if resulting_candidates is not None:
                xdata_candidates = v_points_candidates['V_L']
                ydata_candidates = v_points_candidates['V_M']
                zdata_candidates = v_points_candidates['V_R']
                ax.scatter3D(xdata_candidates, ydata_candidates, zdata_candidates, c='none', edgecolor='white', s=100,
                             label='candidates for next stage')
            if plot_max:
                ax.scatter3D(xdata[max_score_idx], ydata[max_score_idx], zdata[max_score_idx], s=100, c='white', marker='x',
                             label='highest score')
            # butch = ax.scatter(1.350, 0.66, 1.02, marker='x', label='old qubit loc')
            if bounds is not None:
                ax.scatter(*lower_bound, marker='*', label='lower bound')
                ax.scatter(*upper_bound, marker='*', label='upper bound')

            plt.colorbar(im, ax=ax)
            ax.view_init(elev, azim)

            plt.title(f"Perspective {i}: elev={elev}, azim={azim}")
            plt.tight_layout()
        plt.legend()
        # plt.show()
    return fig

def draw_boxes_and_preds(
    _img: np.ndarray,
    _img_blocked,
    locations: np.ndarray,
    predictions=None,
    sidelength: int = 20,
    classification_thresh=0.5,
    marker_offset=5,
    name = "PSB predictions via neural networks"
) -> np.ndarray:
    """
    Function to draw a single box on an image in pixel space.

    Parameters:
    _img (np.ndarray): The image on which the box is to be drawn.
    box (np.ndarray): The coordinates of the box to be drawn.
    condition (str): Determines the color of the box. If 'positive bias', the color is set to the maximum color in the image. Otherwise, it's set to the minimum color in the image.

    Returns:
    np.ndarray: The image with the box drawn on it.
    """
    img = _img.copy()
    img_blocked = _img_blocked.copy()

    for location, pred in zip(locations, predictions):
        start = location - sidelength // 2
        end = location + sidelength // 2
        # print(f'start {start}')
        # print(f'end {end}')
        rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
        if pred is None:
            img[rr, cc] = np.mean(img)
            img_blocked[rr, cc] = np.mean(img_blocked)
        elif pred > classification_thresh:
            img[rr, cc] = np.max(img)
            img_blocked[rr, cc] = np.max(img_blocked)
        elif pred < classification_thresh:
            img[rr, cc] = np.percentile(img, 90)
            img_blocked[rr, cc] = np.percentile(img_blocked, 90)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img, origin="lower", cmap=colormap)
    for location, pred in zip(locations, predictions):
        if pred is not None:
            if pred > classification_thresh:
                color = "green"
            else:
                color = "red"
            # print(f"locaation{location[1], location[0]}")
            if 'psb' in name.lower():
                axs[0].annotate(
                    np.round(pred[0], 4),
                    (location[1] - marker_offset, location[0] + marker_offset),
                    c=color,
                )
            else:
                axs[0].annotate(
                    np.round(pred, 4),
                    (location[1] - marker_offset, location[0] + marker_offset),
                    c=color,
                )
    axs[0].set_title("high B (leaky)")

    axs[1].imshow(img_blocked, origin="lower", cmap=colormap)
    axs[1].set_title("low B (blocked)")
    axs[0].grid(False)
    axs[1].grid(False)
    plt.tight_layout()
    plt.suptitle(name)
    return fig


def draw_box(axs, _img, location, sidelength):
    img = _img.copy()

    start = location - sidelength // 2
    end = location + sidelength // 2
    # print(f'start {start}')
    # print(f'end {end}')
    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    img[rr, cc] = np.max(img)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[1].imshow(img, origin="lower", cmap=colormap)
    axs[1].set_title("overview")

    axs[1].grid(False)
    return axs


def ramp_magnet_before_msmt(ips, magnetic_field):
    ramp_rate = ips.sweeprate_field.get()  # T/min
    previous_field = ips._get_field_setpoint()  # T
    ramp_time = abs(previous_field - magnetic_field) / ramp_rate  # min
    logging.info(
        f"Ramping magnet from {previous_field} T to {magnetic_field} T and waiting {ramp_time} min"
    )
    ips.run_to_field(magnetic_field)
    sleep(ramp_time * 60)  # s


def start_pulsing_tailored(awg, cpg_list, bias_direction, t_burst=4e-9, dead_burst_time=5e-9):
    cb_ro_time = dead_burst_time + t_burst
    pp = PulseParameter(
        t_RO=cb_ro_time,  # readout part
        t_CB=cb_ro_time,  # coulomb blockade part
        t_ramp=4e-9,
        t_burst=4e-9,
        C_ampl=0,
        I_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
        Q_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
        IQ_delay=19e-9,  #
        f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
        f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
        CP_correction_factor=0.848,
    )  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
    # old compensation from MJC before was 0.386 mV/mV (IGOR NEEDS DIV BY 2, here take total splitting !!),
    # we rechecked based on scan #49 to improve compensation 230428
    # )
    cpg_names = [gate.name for gate in cpg_list]

    if 'V_LP' in cpg_names:
        sign_c_ampl_positive_bias = 1
    else:
        sign_c_ampl_positive_bias = -1

    if bias_direction == "positive_bias":
        pp.C_ampl = sign_c_ampl_positive_bias * 0.025
    elif bias_direction == "negative_bias":
        pp.C_ampl = -1 * sign_c_ampl_positive_bias * 0.025
    else:
        raise NotImplementedError

    pp.t_RO = cb_ro_time
    pp.t_CB = cb_ro_time
    pp.t_ramp = 4e-9
    pp.t_burst = t_burst
    pp.IQ_delay = 19e-9  # delay calibrated with scope
    pp.I_ampl = 0.3
    pp.Q_ampl = 0.3
    init_Rabi(pp, awg, cpg_list)
def start_pulsing(awg, cpg_list, bias_direction, t_burst=4e-9):
    pp = PulseParameter(
        t_RO=41e-9,  # readout part
        t_CB=41e-9,  # coulomb blockade part
        t_ramp=4e-9,
        t_burst=4e-9,
        C_ampl=0,
        I_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
        Q_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
        IQ_delay=19e-9,  #
        f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
        f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
        CP_correction_factor=0.848,
    )  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
    # old compensation from MJC before was 0.386 mV/mV (IGOR NEEDS DIV BY 2, here take total splitting !!),
    # we rechecked based on scan #49 to improve compensation 230428
    cpg_names = [gate.name for gate in cpg_list]
    if 'V_LP' in cpg_names:
        sign_c_ampl_positive_bias = 1
    else:
        sign_c_ampl_positive_bias = -1

    if bias_direction == "positive_bias":
        pp.C_ampl = sign_c_ampl_positive_bias * 0.025
    elif bias_direction == "negative_bias":
        pp.C_ampl = -1 * sign_c_ampl_positive_bias * 0.025
    else:
        raise NotImplementedError

    pp.t_RO = 20e-9
    pp.t_CB = 20e-9
    pp.t_ramp = 4e-9
    pp.t_burst = t_burst
    pp.IQ_delay = 19e-9  # delay calibrated with scope
    pp.I_ampl = 0.3
    pp.Q_ampl = 0.3
    init_Rabi(pp, awg, cpg_list)


def plot_qcodes_line_data(data, name="I_SD"):
    data_accessed = data[name]

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    fig = plt.figure()

    plt.scatter(axes_values[0], data_accessed)
    plt.ylabel(name + " [" + data[name].unit + "]")
    plt.xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    # plt.colorbar(label=name + " [" + data[name].unit + "]")
    # plt.show()
    return fig


def plot_qcodes_data(data, name="I_SD", mark=None, box=None):
    data_accessed = data[name]

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    fig = plt.figure()

    plt.imshow(
        data_accessed,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values[1].min(),
            axes_values[1].max(),
            axes_values[0].min(),
            axes_values[0].max(),
        ],
        cmap=colormap,
    )
    try:
        plt.ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
        plt.xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")
    except: 
        print("! Unable to set axis labels")
    
    if mark is not None:
        plt.scatter([mark[0]], [mark[1]], marker="x", color="white", s=100)
    if box is not None:
        plt.plot(box[:, 0], box[:, 1], c="white")
    try:
        plt.colorbar(label=name + " [" + data[name].unit + "]")
    except: 
        print("! Unable to set axis labels")
    # plt.show()
    return fig


def plot_qcodes_with_overview(
    data,
    overview_data,
    location,
    sidelength,
    name="I_SD",
    transpose_plot=False,
    mark=None,
):
    data_accessed = data[name]
    overview_data_accessed = overview_data[name]
    img = overview_data_accessed.to_numpy().copy()

    start = location - sidelength // 2
    end = location + sidelength // 2
    # print(f'start {start}')
    # print(f'end {end}')
    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    if transpose_plot:
        im0 = axs[0].imshow(
            data_accessed.T,
            origin="lower",
            aspect="auto",
            extent=[
                axes_values[0].min(),
                axes_values[0].max(),
                axes_values[1].min(),
                axes_values[1].max(),
            ],
            cmap=colormap,
        )
        axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
        axs[0].set_ylabel(axes_values_names[1] + " [" + axes_units[1] + "]")
        fig.colorbar(im0, ax=axs[0], label=name + " [" + data[name].unit + "]")
        axs[1].imshow(
            img.T,
            origin="lower",
            aspect="auto",
            extent=[
                axes_values_overview[0].min(),
                axes_values_overview[0].max(),
                axes_values_overview[1].min(),
                axes_values_overview[1].max(),
            ],
            cmap=colormap,
        )
        axs[1].set_xlabel(
            axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
        )
        axs[1].set_ylabel(
            axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
        )
        axs[1].set_title("overview")

        axs[1].grid(False)

    else:
        im0 = axs[0].imshow(
            data_accessed,
            origin="lower",
            aspect="auto",
            extent=[
                axes_values[1].min(),
                axes_values[1].max(),
                axes_values[0].min(),
                axes_values[0].max(),
            ],
            cmap=colormap,
        )
        axs[0].set_ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
        axs[0].set_xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")
        fig.colorbar(im0, ax=axs[0], label=name + " [" + data[name].unit + "]")
        if mark is not None:
            axs[0].plot([mark[0]], [mark[1]], "x", color="white")
            axs[0].scatter([mark[0]], [mark[1]], marker="x", color="white", s=100)
        im1 = axs[1].imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[
                axes_values_overview[1].min(),
                axes_values_overview[1].max(),
                axes_values_overview[0].min(),
                axes_values_overview[0].max(),
            ],
            cmap=colormap,
        )
        axs[1].set_ylabel(
            axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
        )
        axs[1].set_xlabel(
            axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
        )

        axs[1].set_title("overview")
        fig.colorbar(im1, ax=axs[1], label=name + " [" + data[name].unit + "]")
        axs[1].grid(False)
    plt.tight_layout()
    # plt.show()
    return fig


def plot_psb_qcodes_with_overview(
    high_B_msmt,
    low_B_msmt,
    overview_data,
    location,
    sidelength,
    name="I_SD"
):
    high_B_data_accessed = high_B_msmt[name]
    overview_data_accessed = overview_data[name]
    low_B_data_accessed = low_B_msmt[name]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = low_B_data_accessed.to_numpy().copy()

    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(high_B_msmt.dims).items():
        axes_values.append(high_B_msmt[item].to_numpy())
        axes_values_names.append(high_B_msmt[item].long_name)
        axes_units.append(high_B_msmt[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(low_B_msmt.dims).items():
        axes_values_additional.append(low_B_msmt[item].to_numpy())
        axes_values_names_additional.append(low_B_msmt[item].long_name)
        axes_units_additional.append(low_B_msmt[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].imshow(
        high_B_data_accessed,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values[1].min(),
            axes_values[1].max(),
            axes_values[0].min(),
            axes_values[0].max(),
        ],
        cmap=colormap,
    )
    axs[0].set_ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")
    axs[0].set_title("high_B_data")
    fig.colorbar(im1, ax=axs[0], label=name + " [" + high_B_msmt[name].unit + "]")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )
    axs[1].set_title("low_B_msmt")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + high_B_msmt[name].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name + " [" + high_B_msmt[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig


#
# def plot_lockin_line_qcodes(
#     line_scan,
#     name_x="LIX",
#     name_y="LIY",
# ):
#     line_scant_x = line_scan[name_x].to_numpy().copy()
#     line_scan_y = line_scan[name_y].to_numpy().copy()
#
#     import sys
#
#     sys.path.append("..")
#     from helper_functions.pca import PCA
#
#     line_scan_accessed = PCA(line_scant_x, line_scan_y)
#
#     axes_values = []
#     axes_values_names = []
#     axes_units = []
#     for item, n in dict(line_scan.dims).items():
#         axes_values.append(line_scan[item].to_numpy())
#         axes_values_names.append(line_scan[item].long_name)
#         axes_units.append(line_scan[item].unit)
#
#     fig= plt.figure(figsize=(12, 4))
#
#     plt.plot(axes_values[0], line_scan_accessed)
#     # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
#     plt.xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
#     plt.ylabel("PCA of lockin")
#     plt.title("EDSR scan")
#
#     plt.tight_layout()
#     # plt.show()
#     return fig



def plot_lockin_line_qcodes(
    line_scan,
    detuning_line_points,
    pulsed_data,
    name="I_SD",
    name_x="LIX",
    name_y="LIY",
):
    line_scant_x = line_scan[name_x].to_numpy().copy()
    line_scan_y = line_scan[name_y].to_numpy().copy()

    import sys

    sys.path.append("..")
    from helper_functions.pca import PCA

    line_scan_accessed = PCA(line_scant_x, line_scan_y)

    pulsed_data_accessed = pulsed_data[name]

    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].plot(axes_values[0], line_scan_accessed)
    # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel("PCA of lockin")
    axs[0].set_title("EDSR scan")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color='white', marker="x", s=100)
    axs[1].set_title("pulsed data")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + pulsed_data[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    return fig


def plot_lockin_line_qcodes_with_overview(
    line_scan,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name="I_SD",
    name_x="LIX",
    name_y="LIY",
):
    line_scant_x = line_scan[name_x].to_numpy().copy()
    line_scan_y = line_scan[name_y].to_numpy().copy()

    import sys

    sys.path.append("..")
    from helper_functions.pca import PCA

    line_scan_accessed = PCA(line_scant_x, line_scan_y)

    overview_data_accessed = overview_data[name]
    pulsed_data_accessed = pulsed_data[name]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points
    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].plot(axes_values[0], line_scan_accessed)
    # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel("PCA of lockin")
    axs[0].set_title("EDSR scan")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color='white', marker="x", s=100)
    axs[1].set_title("pulsed data")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + overview_data[name].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name + " [" + overview_data[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig



def plot_lines_averaged_qcodes_with_overview(
    averaged_lines,
    line_scan,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name="I_SD",
):
    line_scan_accessed = line_scan[name]
    overview_data_accessed = overview_data[name]
    pulsed_data_accessed = pulsed_data[name]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points
    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].plot(axes_values[0], averaged_lines, color='white', marker="x", s=100)
    # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel(name)
    axs[0].set_title("averaged detuning line scans")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color='white', marker="x", s=100)
    axs[1].set_title("triangle")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + overview_data[name].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name + " [" + overview_data[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig

def plot_rabi_oscillations_with_overview(
    line_scan,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
):
    line_scan_x = line_scan['LIX'].to_numpy().copy()
    line_scan_y = line_scan['LIY'].to_numpy().copy()


    name='I_SD'
    line_scan_accessed = PCA(line_scan_x, line_scan_y)
    overview_data_accessed = overview_data[name]
    pulsed_data_accessed = pulsed_data[name]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points
    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].plot(
        axes_values[0], line_scan_accessed
    )
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel('PCA of lockin')
    axs[0].set_title("Rabi oscillations")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color="white", marker="x", s=100)
    axs[1].plot(lp, rp, color="white", linestyle='-')
    axs[1].set_title("pulsed data")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + overview_data[name].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name + " [" + overview_data[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig
def plot_line_qcodes_with_overview(
    line_scan,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name="I_SD",
):
    line_scan_accessed = line_scan[name]
    overview_data_accessed = overview_data[name]
    pulsed_data_accessed = pulsed_data[name]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points
    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].plot(
        axes_values[0], line_scan_accessed
    )
    # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel(name)
    axs[0].set_title("detuning line scan")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color="white", marker="x", s=100)
    axs[1].plot(lp, rp, color="white", linestyle='-')
    axs[1].set_title("pulsed data")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + overview_data[name].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name + " [" + overview_data[name].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig

def plot_line_qcodes_without_overview(
    line_scan,
    detuning_line_points,
    pulsed_data,
    name="I_SD",
):
    line_scan_accessed = line_scan[name]
    pulsed_data_accessed = pulsed_data[name]

    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points


    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(line_scan.dims).items():
        axes_values.append(line_scan[item].to_numpy())
        axes_values_names.append(line_scan[item].long_name)
        axes_units.append(line_scan[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].plot(
        axes_values[0], line_scan_accessed
    )
    # axs[0].set_ylabel(axes_values_names[0] + ' [' + axes_units[0] + ']')
    axs[0].set_xlabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_ylabel(name)
    axs[0].set_title("detuning line scan")

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color="white", marker="x", s=100)
    axs[1].plot(lp, rp, color="white", linestyle='-')
    axs[1].set_title("pulsed data")
    fig.colorbar(im3, ax=axs[1], label=name + " [" + pulsed_data[name].unit + "]")
    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig



def plot_lockin_qcodes_with_overview(
    b_v_det,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name_isd="I_SD",
    name_x="LIX",
    name_y="LIY",
    plot_isd=False,
):
    if plot_isd:
        b_v_det_accessed = b_v_det[name_isd].to_numpy().copy()
    else:
        b_v_det_x = b_v_det[name_x].to_numpy().copy()
        b_v_det_y = b_v_det[name_y].to_numpy().copy()

        import sys

        sys.path.append("..")
        from helper_functions.pca import PCA

        b_v_det_accessed = PCA(b_v_det_x, b_v_det_y)

    overview_data_accessed = overview_data[name_isd]
    pulsed_data_accessed = pulsed_data[name_isd]

    img = overview_data_accessed.to_numpy().copy()
    additional_img = pulsed_data_accessed.to_numpy().copy()
    [rp, lp] = detuning_line_points
    start = location - sidelength // 2
    end = location + sidelength // 2

    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    # img[rr, cc] = np.min(img)
    img[rr, cc] = (np.min(img) + np.max(img)) / 2

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(b_v_det.dims).items():
        axes_values.append(b_v_det[item].to_numpy())
        axes_values_names.append(b_v_det[item].long_name)
        axes_units.append(b_v_det[item].unit)

    axes_values_additional = []
    axes_values_names_additional = []
    axes_units_additional = []
    for item, n in dict(pulsed_data.dims).items():
        axes_values_additional.append(pulsed_data[item].to_numpy())
        axes_values_names_additional.append(pulsed_data[item].long_name)
        axes_units_additional.append(pulsed_data[item].unit)

    axes_values_overview = []
    axes_values_names_overview = []
    axes_units_overview = []
    for item, n in dict(overview_data.dims).items():
        axes_values_overview.append(overview_data[item].to_numpy())
        axes_values_names_overview.append(overview_data[item].long_name)
        axes_units_overview.append(overview_data[item].unit)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axs[0].imshow(
        b_v_det_accessed,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values[1].min(),
            axes_values[1].max(),
            axes_values[0].min(),
            axes_values[0].max(),
        ],
        cmap=colormap,
    )
    axs[0].set_ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
    axs[0].set_xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")
    # axs[0].set_ylabel(name)
    if plot_isd:
        axs[0].set_title("ISD")
        fig.colorbar(im1, ax=axs[0], label=name_isd + " [" + overview_data[name_isd].unit + "]")
    else:
        axs[0].set_title("PCA of lockin")
        fig.colorbar(im1, ax=axs[0], label="PCA")
    # fig.colorbar(im1, ax=axs[0])

    im3 = axs[1].imshow(
        additional_img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_additional[1].min(),
            axes_values_additional[1].max(),
            axes_values_additional[0].min(),
            axes_values_additional[0].max(),
        ],
        cmap=colormap,
    )
    axs[1].set_ylabel(
        axes_values_names_additional[0] + " [" + axes_units_additional[0] + "]"
    )
    axs[1].set_xlabel(
        axes_values_names_additional[1] + " [" + axes_units_additional[1] + "]"
    )

    axs[1].scatter(lp, rp, color="white", marker="x", s=100)
    axs[1].set_title("high res scan")
    fig.colorbar(im3, ax=axs[1], label=name_isd + " [" + overview_data[name_isd].unit + "]")

    im2 = axs[2].imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values_overview[1].min(),
            axes_values_overview[1].max(),
            axes_values_overview[0].min(),
            axes_values_overview[0].max(),
        ],
        cmap=colormap,
    )
    axs[2].set_ylabel(
        axes_values_names_overview[0] + " [" + axes_units_overview[0] + "]"
    )
    axs[2].set_xlabel(
        axes_values_names_overview[1] + " [" + axes_units_overview[1] + "]"
    )
    axs[2].set_title("overview")
    fig.colorbar(im2, ax=axs[2], label=name_isd + " [" + overview_data[name_isd].unit + "]")

    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig


def plot_danon_qcodes_with_overview(
    b_v_det,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name_isd="I_SD",
    name_x="LIX",
    name_y="LIY",
    plot_isd=True,
):
    return plot_lockin_qcodes_with_overview(
        b_v_det,
        detuning_line_points,
        pulsed_data,
        overview_data,
        location,
        sidelength,
        name_isd=name_isd,
        name_x=name_x,
        name_y=name_y,
        plot_isd=plot_isd,
    )


def plot_b_v_det_qcodes_with_overview(
    b_v_det,
    detuning_line_points,
    pulsed_data,
    overview_data,
    location,
    sidelength,
    name_isd="I_SD",
    name_x="LIX",
    name_y="LIY",
):
    return plot_lockin_qcodes_with_overview(
        b_v_det,
        detuning_line_points,
        pulsed_data,
        overview_data,
        location,
        sidelength,
        name_isd=name_isd,
        name_x=name_x,
        name_y=name_y,
    )
