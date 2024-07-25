import pickle

import numpy as np
from matplotlib import pyplot as plt
from qcodes import load_by_guid
from treelib import Node, Tree
import os

import sys
sys.path.append('../') #so that 'pipelines' is in scope

def get_measurement_time_of_candidate(time_dict):
    return time_dict["measurement_end"][1] - time_dict["measurement_start"][1]


def get_analysis_time_of_candidate(time_dict):
    return time_dict["analysis_end"][1] - time_dict["analysis_start"][1]


def get_total_time_of_candidate(time_dict):
    msmt_time = get_measurement_time_of_candidate(time_dict)
    analysis_time = get_analysis_time_of_candidate(time_dict)
    return msmt_time + analysis_time


def build_tree(tree, path, parent=None):
    node_name = os.path.basename(path) if parent else path
    # print(f'node_name {node_name}, path {path}, parent {parent}')
    # tree.create_node(node_name, path, parent=parent)
    names = os.listdir(path)
    name_pickle = [name for name in names if ".pkl" in name][0]
    pickle_path = os.path.join(path, name_pickle)
    with open(pickle_path, "rb") as f:
        candidate = pickle.load(f)
    name_of_stage = candidate.receiver
    index_of_stage = candidate.sender_candidate_id
    name_of_node = name_of_stage + "_" + str(index_of_stage)
    # print(name_of_node)
    tree.create_node(tag=name_of_node, identifier=path, parent=parent, data=candidate)
    for item in os.listdir(path):
        # print(f'item {item}')
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # print(f'building tree')
            build_tree(tree, item_path, parent=path)


def total_time_layer(tree, depth):
    time_dict_list = [
        node.data.times
        for node in tree.all_nodes()
        if tree.depth(node) == depth and node.data is not None
    ]
    time_list = [get_total_time_of_candidate(time_dict) for time_dict in time_dict_list]
    return np.sum(time_list)


def time_per_candidate_in_layer(tree, depth):
    time_dict_list = [
        node.data.times
        for node in tree.all_nodes()
        if tree.depth(node) == depth and node.data is not None
    ]
    time_list = [get_total_time_of_candidate(time_dict) for time_dict in time_dict_list]
    return time_list


def partial_time_layer(tree, depth, part="measurement"):
    time_dict_list = [
        node.data.times
        for node in tree.all_nodes()
        if tree.depth(node) == depth and node.data is not None
    ]
    if part == "measurement":
        time_list = [
            get_measurement_time_of_candidate(time_dict) for time_dict in time_dict_list
        ]
    elif part == "analysis":
        time_list = [
            get_analysis_time_of_candidate(time_dict) for time_dict in time_dict_list
        ]
    return np.sum(time_list)


def get_name_depth(tree, depth):
    nodes = [
        node.tag
        for node in tree.all_nodes()
        if tree.depth(node) == depth and node.data is not None
    ]
    return nodes[0][:-2]


def get_all_time_per_stage(tree):
    depth = tree.depth() + 1
    time_dict = {}
    for i in range(depth):
        name_of_stage = get_name_depth(tree, i)
        time_dict[name_of_stage] = {"time": (total_time_layer(tree, i)), "depth": i}

    return time_dict


def get_all_time_per_stage_per_candidate(tree):
    depth = tree.depth() + 1
    time_dict = {}
    for i in range(depth):
        name_of_stage = get_name_depth(tree, i)
        time_dict[name_of_stage] = {
            "time": (time_per_candidate_in_layer(tree, i)),
            "depth": i,
        }

    return time_dict


def get_partial_time_per_stage(tree):
    depth = tree.depth() + 1
    time_dict = {}
    for i in range(depth):
        name_of_stage = get_name_depth(tree, i)
        time_dict[name_of_stage] = {
            "time_measurement": (partial_time_layer(tree, i)),
            "time_analysis": (partial_time_layer(tree, i, part="analysis")),
            "depth": i,
        }

    return time_dict


short_to_long_name = {
    "HSBldr": "Hypersurface Builder",
    "DDFndr": "Double Dot Finder",
    "BSO": "Base Separation Optimisation",
    "Wdscn": "PSB via wide scan",
    "Cntr": "Centering",
    "HRPSBClf": "High Resolution PSB",
    "Danon": "Danon Gap Check",
    "ROOpt": "Readout optimisation",
    "EChck": "EDSR check",
    "EDSRline": "EDSR Spectroscopy",
    "Rabi": "Rabi chevron",
    "RabiO": "Rabi oscillations",
}


def plot_time_per_candidate_and_stage(time_dict):
    fig, ax = plt.subplots()
    for key, value in time_dict.items():
        display_name = short_to_long_name[key]
        bottom = 0
        for value in value["time"]:
            ax.bar(display_name, value / 60 / 60, label=key, bottom=bottom)
            bottom += value / 60 / 60
        # ax.bar(short_to_long_name[key], value['time']/60/60, label=key, color='red')
    ax.set_ylabel("time [h]")
    ax.set_xlabel("stage")
    # turn the x-labels 90 degrees
    plt.xticks(rotation=90)
    # ax.legend()
    plt.show()


def plot_time_parts_per_stage(time_dict):
    fig, ax = plt.subplots()
    for key, value in time_dict.items():
        ax.bar(
            short_to_long_name[key],
            value["time_measurement"] / 60 / 60,
            label="measurement",
            color="red",
        )
        ax.bar(
            short_to_long_name[key],
            value["time_analysis"] / 60 / 60,
            label="analysis",
            color="orange",
            bottom=value["time_measurement"] / 60 / 60,
        )
    ax.set_ylabel("time [h]")
    ax.set_xlabel("stage")
    # turn the x-labels 90 degrees
    plt.xticks(rotation=90)
    # ax.legend()
    plt.show()


# def plot_time_per_stage(time_dict):
#     fig, ax = plt.subplots()
#     for key, value in time_dict.items():
#         ax.bar(short_to_long_name[key], value['time'] / 60 / 60, label=key, color='red')
#     ax.set_ylabel('time [h]')
#     ax.set_xlabel('stage')
#     # turn the x-labels 90 degrees
#     plt.xticks(rotation=90)
#     # ax.legend()
#     plt.show()
def plot_time_per_stage(time_dict_tree, tree_descriptions):
    fig, ax = plt.subplots()

    n_trees = len(time_dict_tree)
    bar_width = 0.8 / n_trees  # to accommodate all trees bars in a width of 0.8
    stages = list(short_to_long_name.keys())
    colors = plt.cm.gnuplot(np.linspace(0, 1, n_trees))

    for idx_tree, (time_dict, description) in enumerate(
        zip(time_dict_tree, tree_descriptions)
    ):
        for idx_stage, (key, value) in enumerate(time_dict.items()):
            position = idx_stage - 0.4 + (bar_width * idx_tree)  # setting the position
            ax.bar(
                position,
                value["time"] / 60 / 60,
                width=bar_width,
                label=description if idx_stage == 0 else "",
                color=colors[idx_tree],
            )

    ax.set_ylabel("time [h]")
    ax.set_xlabel("stage")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([short_to_long_name[stage] for stage in stages], rotation=90)
    ax.legend()


grouped_stages = {
    "Define DQD": ["HSBldr", "DDFndr"],
    "Tune barriers": ["BSO"],
    "Find PSB": ["Wdscn", "Cntr", "HRPSBClf", "Danon"],
    "Find readout": ["ROOpt", "EChck", "EDSRline", "Rabi", "RabiO"],
}


def get_grouped_times(tree):
    granular_times = get_all_time_per_stage(tree)
    # print(granular_times)
    grouped_times = {}
    for i, (group_name, stages) in enumerate(grouped_stages.items()):
        grouped_times[group_name] = {"time": 0, "depth": i}
        for stage in stages:
            grouped_times[group_name]["time"] += granular_times[stage]["time"]
    return grouped_times


def plot_time_per_stage_grouped(time_dict_tree, tree_descriptions):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    n_trees = len(time_dict_tree)
    bar_width = 0.8 / n_trees  # to accommodate all trees bars in a width of 0.8
    stages = list(grouped_stages.keys())
    colors = plt.cm.plasma(np.linspace(0, 1, n_trees))
    colors = plt.cm.winter(np.linspace(0, 1, n_trees))
    colors = plt.cm.spring(np.linspace(0, 1, n_trees))

    for idx_tree, (time_dict, description) in enumerate(
        zip(time_dict_tree, tree_descriptions)
    ):
        for idx_stage, (key, value) in enumerate(time_dict.items()):
            position = idx_stage - 0.4 + (bar_width * idx_tree)  # setting the position
            ax.bar(
                position,
                value["time"] / 60 / 60,
                width=bar_width,
                label=description if idx_stage == 0 else "",
                color=colors[idx_tree],
            )

    ax.set_ylabel("Time (h)")
    ax.set_xlabel("Stage")
    ax.set_xticks(range(len(stages)))
    ax.tick_params(direction="in")
    ax.set_xticklabels([stage for stage in stages])  # , rotation=60)
    ax.xaxis.set_ticks_position("none")
    ax.legend()
    plt.tight_layout()
    return fig  # plt.show()


def get_total_time(times_dict_trees):
    times = []
    for times_dict in times_dict_trees:
        total_time = 0
        for key, value in times_dict.items():
            total_time += value["time"]
        times.append(total_time)
    return times


def plot_time_per_stage_grouped_fail_and_success(
    time_dict_tree, tree_descriptions, time_dict_trees_failed, colors=None
):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    n_trees = len(time_dict_tree) + len(time_dict_trees_failed) + 1
    n_trees_success = len(time_dict_tree)
    bar_width = 0.8 / n_trees  # to accommodate all trees bars in a width of 0.8
    stages = list(grouped_stages.keys())

    if colors is None:
        colors = plt.cm.plasma(np.linspace(0, 1, n_trees_success))
        colors = plt.cm.winter(np.linspace(0, 1, n_trees_success))
        colors = plt.cm.brg(np.linspace(0, 1, n_trees_success))

    for idx_tree, (time_dict, description) in enumerate(
        zip(time_dict_tree, tree_descriptions)
    ):
        for idx_stage, (key, value) in enumerate(time_dict.items()):
            position = idx_stage - 0.4 + (bar_width * idx_tree)  # setting the position
            ax.bar(
                position,
                value["time"] / 60 / 60,
                width=bar_width,
                label=description if idx_stage == 0 else "",
                color=colors[idx_tree],
            )
    for idx_tree_failed, time_dict in enumerate(time_dict_trees_failed):
        description = "failed"
        color = "tab:red"
        for idx_stage, (key, value) in enumerate(time_dict.items()):
            position = (
                idx_stage - 0.4 + (bar_width * (idx_tree + idx_tree_failed + 2))
            )  # setting the position
            ax.bar(
                position,
                value["time"] / 60 / 60,
                width=bar_width,
                label=description if (idx_stage == 0 and idx_tree_failed == 0) else "",
                color=color,
            )

    ax.set_ylabel("Time (h)")
    ax.set_xlabel("Stage")
    ax.set_xticks(range(len(stages)))
    ax.tick_params(direction="in")
    ax.set_xticklabels([stage for stage in stages])  # , rotation=60)
    ax.xaxis.set_ticks_position("none")
    # ax.legend()
    plt.tight_layout()
    return fig  # plt.show()


def get_total_time(times_dict_trees):
    times = []
    for times_dict in times_dict_trees:
        total_time = 0
        for key, value in times_dict.items():
            total_time += value["time"]
        times.append(total_time)
    return times


def plot_coarse_tuning_results(
    all_results_list,
    resulting_candidates_list=None,
    wide_scan_dones_list=None,
    ro_performed_list=None,
    qubits_found_list=None,
    color_list=["white"],
    major_axis_poff=None,
    plot_all=False,
    plot_candidates=False,
):
    # print(len(all_results_list), len(resulting_candidates_list), len(wide_scan_list), len(qubits_found_list), len(color_list))

    # plt.show()

    fig = plt.figure(figsize=(5.8, 2.99))
    axs = []
    perspectives = [("V_L", "V_M"), ("V_L", "V_R")]
    for i, (x_name, y_name) in enumerate(perspectives):
        ax = fig.add_subplot(1, 2, i + 1)
        axs.append(ax)
    print(f"Plotting {len(all_results_list)} results")
    for (
        all_results,
        resulting_candidates,
        wide_scan_dones,
        ro_performed,
        qubits_found,
        color,
    ) in zip(
        all_results_list,
        resulting_candidates_list,
        wide_scan_dones_list,
        ro_performed_list,
        qubits_found_list,
        color_list,
    ):
        print(f"color: {color}")
        if resulting_candidates is not None:
            v_points_candidates = {"V_L": [], "V_R": [], "V_M": []}
            for candidate in resulting_candidates:
                v_points_candidates["V_L"].append(candidate.info["config"]["V_L"])
                v_points_candidates["V_M"].append(candidate.info["config"]["V_M"])
                v_points_candidates["V_R"].append(candidate.info["config"]["V_R"])
            if major_axis_poff is not None:
                v_points_candidates["V_L"] /= major_axis_poff[0]
                v_points_candidates["V_M"] /= major_axis_poff[1]
                v_points_candidates["V_R"] /= major_axis_poff[2]

        if wide_scan_dones is not None:
            v_points_wide_scan = {"V_L": [], "V_R": [], "V_M": []}
            for ws in wide_scan_dones:
                v_points_wide_scan["V_L"].append(ws.info["config"]["V_L"])
                v_points_wide_scan["V_M"].append(ws.info["config"]["V_M"])
                v_points_wide_scan["V_R"].append(ws.info["config"]["V_R"])
            if major_axis_poff is not None:
                v_points_wide_scan["V_L"] /= major_axis_poff[0]
                v_points_wide_scan["V_M"] /= major_axis_poff[1]
                v_points_wide_scan["V_R"] /= major_axis_poff[2]
        if ro_performed is not None:
            v_points_ro_performed = {"V_L": [], "V_R": [], "V_M": []}
            for ro in ro_performed:
                v_points_ro_performed["V_L"].append(ro.info["config"]["V_L"])
                v_points_ro_performed["V_M"].append(ro.info["config"]["V_M"])
                v_points_ro_performed["V_R"].append(ro.info["config"]["V_R"])
            if major_axis_poff is not None:
                v_points_ro_performed["V_L"] /= major_axis_poff[0]
                v_points_ro_performed["V_M"] /= major_axis_poff[1]
                v_points_ro_performed["V_R"] /= major_axis_poff[2]

        if qubits_found is not None:
            v_points_qubits_found = {"V_L": [], "V_R": [], "V_M": []}
            for qubit in qubits_found:
                v_points_qubits_found["V_L"].append(qubit.info["config"]["V_L"])
                v_points_qubits_found["V_M"].append(qubit.info["config"]["V_M"])
                v_points_qubits_found["V_R"].append(qubit.info["config"]["V_R"])

            if major_axis_poff is not None:
                v_points_qubits_found["V_L"] /= major_axis_poff[0]
                v_points_qubits_found["V_M"] /= major_axis_poff[1]
                v_points_qubits_found["V_R"] /= major_axis_poff[2]

        v_points = {"V_L": [], "V_R": [], "V_M": []}

        cotunneling_detections = []
        for result in all_results:
            v_points["V_L"].append(result.config["V_L"])
            v_points["V_M"].append(result.config["V_M"])
            v_points["V_R"].append(result.config["V_R"])
            cotunneling_detections.append(result.metadata["is_axes_cotunneling"])
            cotunneling_detections[-1]["V_M"] = False

        if major_axis_poff is not None:
            v_points["V_L"] /= major_axis_poff[0]
            v_points["V_M"] /= major_axis_poff[1]
            v_points["V_R"] /= major_axis_poff[2]

        for i, (x_name, y_name) in enumerate(perspectives):
            ax = axs[i]
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            xdata = v_points[x_name]
            ydata = v_points[y_name]
            ax.set_xlabel(f"{x_name} (fraction of pinch off)")
            ax.set_ylabel(f"{y_name} (fraction of pinch off)")
            if plot_all:
                im = ax.scatter(
                    xdata, ydata, c=color, label="investigated by coarse tuning"
                )
            if resulting_candidates is not None and plot_candidates:
                xdata_candidates = v_points_candidates[x_name]
                ydata_candidates = v_points_candidates[y_name]
                ax.scatter(
                    xdata_candidates,
                    ydata_candidates,
                    c="none",
                    edgecolor="white",
                    s=100,
                    label="candidates for next stage",
                )
            if wide_scan_dones is not None:
                xdata_wide_scan = v_points_wide_scan[x_name]
                ydata_wide_scan = v_points_wide_scan[y_name]
                ax.scatter(
                    xdata_wide_scan,
                    ydata_wide_scan,
                    color=color,
                    s=10,
                    label="Checked for PSB",
                )
            if ro_performed is not None:
                xdata_ro_performed = v_points_ro_performed[x_name]
                ydata_ro_performed = v_points_ro_performed[y_name]
                ax.scatter(
                    xdata_ro_performed,
                    ydata_ro_performed,
                    c="none",
                    edgecolor=color,
                    s=75,
                    linestyle="--",
                    label="PSB found",
                )

            if qubits_found is not None:
                xdata_qubits_found = v_points_qubits_found[x_name]
                ydata_qubits_found = v_points_qubits_found[y_name]
                ax.scatter(
                    xdata_qubits_found,
                    ydata_qubits_found,
                    c="none",
                    edgecolor=color,
                    linestyle="-",
                    s=150,
                    label="Rabi found",
                )
            ax.set_xlim(0.89, 1)
            ax.set_ylim(0.7, 1)
        plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys())
    axs[0].tick_params(direction="in")
    axs[1].tick_params(direction="in")
    leg = axs[1].get_legend()
    leg.legendHandles[0].set_color("black")
    leg.legendHandles[1].set_edgecolor("black")
    leg.legendHandles[2].set_edgecolor("black")
    return fig


from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_g_factor(magnetic_field, frequency):
    mu_B = 9.274e-24
    h = 6.626e-34
    # h_bar = h/(2*np.pi)
    return h / mu_B * frequency / magnetic_field


from scipy.optimize import curve_fit


def rabi_oscillations(t, A, omega, tau, phase, offset, decay_time, decay_offset):
    """Function for Rabi oscillations."""
    return (
        A * np.exp(-t / tau) * np.cos(omega * t + phase)
        + offset
        + np.exp(-t / decay_time) * decay_offset
    )


def fit_rabi_oscillations(t, signal, burst_time=1e-8):
    """Fit Rabi oscillations."""
    # Initial guesses for A, omega, tau, offset
    # guess = [max(signal) - min(signal), 2.0*np.pi/np.mean(np.diff(t)), np.mean(t), np.mean(signal)]

    # t = data['t_burst'].to_numpy()
    A = 1.0
    omega = 2 * np.pi * 1 / burst_time
    tau = 1e-8
    offset = 0.5
    phase = np.pi / 4
    decay_time = tau
    decay_offset = 0.5

    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    guess = [A, omega, tau, phase, offset, decay_time, decay_offset]

    # Curve fitting
    popt, pcov = curve_fit(rabi_oscillations, t, signal, p0=guess, maxfev=10000)

    return popt, pcov


def plot_qcodes_data(
    data,
    name="I_SD",
    mark=None,
    box=None,
    colormap="icefire",
    figsize=(2.1, 3),
    loc_colorbar="top",
    bias_direction="positive",
):
    data_accessed = data[name]

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    # fig, ax = plt.subplots(figsize=figsize)
    # Create a figure
    fig = plt.figure(figsize=figsize)

    # Define axes for the main plot (adjust the values as needed)
    if loc_colorbar == "top":
        ax_main = fig.add_axes([0.25, 0.2, 0.7, 0.65])  # left, bottom, width, height
    else:
        ax_main = fig.add_axes([0.25, 0.2, 0.65, 0.7])  # left, bottom, width, height
    data_accessed = -data_accessed * 1e12
    if bias_direction == "negative":
        data_accessed = data_accessed - np.max(data_accessed)
    else:
        data_accessed = data_accessed - np.min(data_accessed)
    im1 = ax_main.imshow(
        data_accessed,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values[1].min() * 1e3,
            axes_values[1].max() * 1e3,
            axes_values[0].min() * 1e3,
            axes_values[0].max() * 1e3,
        ],
        cmap=colormap,
    )
    # try:
    ax_main.set_ylabel(axes_values_names[0] + " (m" + axes_units[0] + ")")
    ax_main.set_xlabel(axes_values_names[1] + " (m" + axes_units[1] + ")")
    # except:
    #     print("! Unable to set axis labels")

    if mark is not None:
        plt.scatter(
            np.array([mark[0]]) * 1e3,
            np.array([mark[1]]) * 1e3,
            marker="x",
            color="white",
            s=100
        )
    if box is not None:
        plt.plot(box[:, 0] * 1e3, box[:, 1] * 1e3, c="white")
    # try:
    # divider = make_axes_locatable(ax)
    # cax = divider.new_vertical(size = '5%', pad = 0.5)
    # fig.add_axes(cax)
    # fig.colorbar(im1, cax = cax, orientation = 'horizontal', shrink = 0.5)
    # cax.xaxis.set_ticks_position("top")
    # plt.colorbar(label=name + " (" + data[name].unit + ")", shrink=0.66)

    if loc_colorbar == "top":
        cbar_ax = fig.add_axes([0.45, 0.875, 0.5, 0.03])  # x, y, width, height
    else:
        cbar_ax = fig.add_axes([0.925, 0.22, 0.03, 0.50])  # x, y, width, height

    # Add a label to the left of the colorbar
    if loc_colorbar == "top":
        # Create the colorbar in the custom axes
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
        # fig.text(0.2, 0.9, "$I_\mathrm{SD}$ (pA)", va='center', ha='center')
        fig.text(0.2, 0.9, "|I| (pA)", va="center", ha="center")
        cbar_ax.xaxis.set_ticks_position("top")
    else:
        cbar = fig.colorbar(im1, cax=cbar_ax)
        # fig.text(0.95, 0.9, "$I_\mathrm{SD}$ (pA)", va='center', ha='center')#, rotation=90)
        fig.text(0.95, 0.9, "|I| (pA)", va="center", ha="center")  # , rotation=90)
    # except:
    #     print("! Unable to set axis labels")
    # plt.show()
    ax_main.tick_params(direction="in")
    plt.tight_layout()
    return fig


def plot_lockin_transposed(
    data,
    name_x="LIX",
    name_y="LIY",
    colormap="icefire",
    figsize=(2.1, 3.3),
    loc_colorbar="top",
    vline=None,
    y_axis_multiplier=1e9,
    invert_current=False,
):
    data_x = data[name_x].to_numpy().copy()
    data_y = data[name_y].to_numpy().copy()

    data_accessed = PCA(data_x, data_y)

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    # fig, ax = plt.subplots(figsize=(3, 3))
    # Create a figure
    fig = plt.figure(figsize=figsize)

    if loc_colorbar == "top":
        ax_main = fig.add_axes([0.25, 0.2, 0.7, 0.65])  # left, bottom, width, height
    else:
        ax_main = fig.add_axes([0.25, 0.2, 0.65, 0.7])  # left, bottom, width, height

    data_accessed = data_accessed * 1e15
    if invert_current:
        data_accessed = -data_accessed
    data_accessed = data_accessed - np.min(data_accessed)
    im1 = ax_main.imshow(
        data_accessed.T,
        origin="lower",
        aspect="auto",
        extent=[
            axes_values[0].min() * 1e3,
            axes_values[0].max() * 1e3,
            axes_values[1].min() * y_axis_multiplier,
            axes_values[1].max() * y_axis_multiplier,
        ],
        cmap=colormap,
    )
    if y_axis_multiplier == 1e9:
        ax_main.set_ylabel(axes_values_names[1] + " (n" + axes_units[1] + ")")
    elif y_axis_multiplier == 1e-9:
        ax_main.set_ylabel(axes_values_names[1] + " (G" + axes_units[1] + ")")
    else:
        ax_main.set_ylabel(axes_values_names[1] + " (" + axes_units[1] + ")")
    ax_main.set_xlabel(axes_values_names[0] + " (m" + axes_units[0] + ")")

    # plt.title("PCA of lockin")
    # divider = make_axes_locatable(ax)
    # cax = divider.new_vertical(size = '5%', pad = 0.2)
    # fig.add_axes(cax)
    # fig.colorbar(im1, cax = cax, orientation = 'horizontal')
    # cax.xaxis.set_ticks_position("top")
    # plt.colorbar(label=" PCA of lock in (pA)", shrink=0.66)
    # fig.colorbar(im1, label="PCA", orientation="horizontal")

    if loc_colorbar == "top":
        # cbar_ax = fig.add_axes([0.4, 0.875, 0.55, 0.03]) # x, y, width, height
        cbar_ax = fig.add_axes([0.45, 0.875, 0.5, 0.03])  # x, y, width, height

        # Create the colorbar in the custom axes
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
        cbar_ax.xaxis.set_ticks_position("top")
        # Add a label to the left of the colorbar
        # fig.text(0.175, 0.9, "$I_\mathrm{SD}$ via\nlock in (fA)", va='center', ha='center')
        fig.text(0.175, 0.9, "|I| (fA)", va="center", ha="center")
    else:
        cbar_ax = fig.add_axes([0.925, 0.22, 0.03, 0.50])  # x, y, width, height
        cbar = fig.colorbar(im1, cax=cbar_ax)
        # fig.text(0.95, 0.9, "$I_\mathrm{SD}$ via\nlock in (fA)", va='center', ha='center')#, rotation=90)
        fig.text(0.95, 0.9, "ILI (fA)", va="center", ha="center")  # , rotation=90)
    if vline is not None:
        ax_main.axvline(vline * 1e3, color="white", linestyle="--")
    ax_main.tick_params(direction="in", color="white")
    plt.grid(False)

    plt.tight_layout()
    # plt.show()
    return fig


def plot_lockin(data, name_x="LIX", name_y="LIY", colormap="icefire"):
    data_x = data[name_x].to_numpy().copy()
    data_y = data[name_y].to_numpy().copy()

    data_accessed = PCA(data_x, data_y)

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(data.dims).items():
        axes_values.append(data[item].to_numpy())
        axes_values_names.append(data[item].long_name)
        axes_units.append(data[item].unit)

    fig = plt.figure(figsize=(2.5, 2.5))
    data_accessed = data_accessed * 1e15
    data_accessed = data_accessed - np.min(data_accessed)
    im1 = plt.imshow(
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
    plt.ylabel(axes_values_names[0] + " (" + axes_units[0] + ")")
    plt.xlabel(axes_values_names[1] + " (" + axes_units[1] + ")")

    # plt.title("PCA of lock in")
    fig.colorbar(im1, label="ILI (fA)")

    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    return fig


from scipy.ndimage import gaussian_filter1d


def plot_line(rabi_osc_state, figsize=(1.5, 1.25), invert_current=False, sigma=1):
    data_accessed = []
    for guid_rabi in rabi_osc_state.data_identifiers["rabi_measurement"]:
        data = load_by_guid(guid_rabi)
        rabi_xarray = data.to_xarray_dataset()
        data_x = rabi_xarray["LIX"].to_numpy().copy()
        data_y = rabi_xarray["LIY"].to_numpy().copy()

        data_accessed.append(PCA(data_x, data_y))

    axes_values = []
    axes_values_names = []
    axes_units = []
    for item, n in dict(rabi_xarray.dims).items():
        axes_values.append(rabi_xarray[item].to_numpy())
        axes_values_names.append(rabi_xarray[item].long_name)
        axes_units.append(rabi_xarray[item].unit)

    averaged_rabi = np.mean(data_accessed, axis=0)
    t = load_by_guid(guid_rabi).to_xarray_dataset()["t_burst"].to_numpy()
    averaged_rabi_for_fit = gaussian_filter1d(averaged_rabi, sigma)
    popt, pcov = fit_rabi_oscillations(t, averaged_rabi_for_fit)
    omega_fitted = popt[1]
    omega_error = np.sqrt(np.diag(pcov))[1]
    rabi_frequency_new_fit = omega_fitted / (2 * np.pi)
    rabi_frequency_error = omega_error / (2 * np.pi)
    print("Rabi frequency: ", rabi_frequency_new_fit / 1e6, " MHz")
    print("Rabi frequency error: ", rabi_frequency_error / 1e6, " MHz")

    fitted_rabi = rabi_oscillations(t, *popt)

    fig, ax_main = plt.subplots(1, 1)
    if invert_current:
        fitted_rabi = -fitted_rabi
        averaged_rabi_for_fit = -averaged_rabi_for_fit
    else:
        averaged_rabi_for_fit = averaged_rabi_for_fit
    fitted_rabi = fitted_rabi - np.min(fitted_rabi)
    averaged_rabi_for_fit = (averaged_rabi_for_fit - np.min(averaged_rabi_for_fit)) / (
        np.max(averaged_rabi_for_fit) - np.min(averaged_rabi_for_fit)
    )
    im1 = plt.plot(
        axes_values[0] * 1e9, averaged_rabi_for_fit, color="black", linewidth=1
    )
    ax_main.plot(t * 1e9, fitted_rabi, "r-")
    plt.ylabel("ILI (fA)")
    plt.xlabel("tburst (ns)")

    print(rabi_osc_state.info["magnetic_field"])

    plt.tight_layout()
    # plt.show()
    ax_main.tick_params(direction="in", color="white")

    plt.show()

    fig, ax_main = plt.subplots(1, 1, figsize=figsize)
    averaged_rabi = averaged_rabi * 1e15
    if invert_current:
        averaged_rabi = -averaged_rabi
    averaged_rabi = averaged_rabi - np.min(averaged_rabi)
    im1 = plt.plot(axes_values[0] * 1e9, averaged_rabi, color="black", linewidth=1)
    plt.ylabel("ILI (fA)")
    plt.xlabel("tburst (ns)")

    print(rabi_osc_state.info["magnetic_field"])

    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    ax_main.tick_params(direction="in", color="white")

    return fig

def PCA(*arrays):
    arrays_copy = [np.copy(array) for array in arrays]
    for array in arrays_copy:
        array[np.isnan(array)] = np.mean(np.ma.masked_invalid(array))

    Z = np.stack(arrays_copy, axis=-1)
    shape = Z.shape

    # summing over every axis except the last
    u = np.mean(Z, axis=tuple(range(0, shape.__len__() - 1)), keepdims=True)

    B = (Z - u).reshape(np.product(shape[0:-1]), shape[-1])
    C = np.einsum("ki, kj -> ij", B, B)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    arg_sorted = np.flip(eigen_values.argsort())
    eigen_vectors = eigen_vectors[:, arg_sorted]
    pca = np.einsum("ik, kj -> ij", B, eigen_vectors).reshape(shape)[..., 0]
    return pca

def plot_qubits(
    spectroscopy_state,
    rabi_chevron_state,
    bias_triangle_state,
    rabi_osc_state,
    name_qubit="first_run",
    bias_direction="positive",
    invert_current_spectroscopy=False,
    invert_current_chevron=False,
    sigma=1,
):
    figure_path = "figures/individual_qubits/" + name_qubit + '/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    spectroscopy_xarray = load_by_guid(
        spectroscopy_state.data_identifiers["edsr_measurement"]
    ).to_xarray_dataset()
    colormap = "vlag"  # seaborn.diverging_palette(145, 300, s=60, as_cmap=True)
    colormap = "icefire"
    fig = plot_lockin(spectroscopy_xarray, colormap=colormap)
    plt.savefig(figure_path + "spectroscopy.pdf")
    plt.show()

    rabi_chevron_xarray = load_by_guid(
        rabi_chevron_state.data_identifiers["rabi_measurement"]
    ).to_xarray_dataset()
    magnetic_fields = rabi_chevron_xarray["IPS_field_setpoint_wait"].to_numpy()
    mag_delta = (np.max(magnetic_fields) - np.min(magnetic_fields)) / 2
    plot_lockin(rabi_chevron_xarray, colormap=colormap)
    plt.savefig(figure_path + "rabi_chevron.pdf")
    plt.show()

    plot_lockin_transposed(
        rabi_chevron_xarray, colormap=colormap, invert_current=invert_current_chevron
    )
    plt.savefig(figure_path + "rabi_chevron_transposed.pdf")
    plt.show()

    plot_lockin_transposed(
        rabi_chevron_xarray,
        colormap=colormap,
        figsize=(3, 2.5),
        loc_colorbar="right",
        invert_current=invert_current_chevron,
    )
    plt.savefig(figure_path + "rabi_chevron_transposed_cbar_right.pdf")
    plt.show()

    high_res_xarray = load_by_guid(
        bias_triangle_state.data_identifiers["high_magnet_measurement"]
    ).to_xarray_dataset()
    plot_qcodes_data(
        high_res_xarray, figsize=(2.25, 2.25), bias_direction=bias_direction
    )
    plt.savefig(figure_path + "triangles.pdf")
    plt.show()

    plot_qcodes_data(
        high_res_xarray,
        figsize=(3, 2.5),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig(figure_path + "triangles_cbar_right.pdf")
    plt.show()

    plot_qcodes_data(
        high_res_xarray,
        figsize=(1.5, 2.3),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig(figure_path + "triangles_high_squeezed.pdf")
    plt.show()

    low_res_xarray = load_by_guid(
        bias_triangle_state.data_identifiers["low_magnet_measurement"]
    ).to_xarray_dataset()

    plot_qcodes_data(
        low_res_xarray,
        figsize=(1.5, 2.3),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig("figures/individual_qubits/" + name_qubit + "/triangles_low_squeezed.pdf")
    plt.show()

    pulsed_xarray = load_by_guid(
        rabi_chevron_state.info["pulsed_msmt_id"]
    ).to_xarray_dataset()
    plot_qcodes_data(pulsed_xarray, figsize=(2.7, 2.7), bias_direction=bias_direction)
    plt.savefig(figure_path + "pulsed_triangles.pdf")
    plt.show()

    plot_qcodes_data(
        pulsed_xarray,
        figsize=(3, 2.5),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig(figure_path + "pulsed_triangles_cbar_right.pdf")
    plt.show()

    plot_qcodes_data(
        high_res_xarray,
        figsize=(1.5, 1.25),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig(figure_path + "small_triangles_cbar_right.pdf")
    plt.show()

    plot_qcodes_data(
        high_res_xarray,
        figsize=(1.5, 1.25),
        loc_colorbar="right",
        bias_direction=bias_direction,
    )
    plt.savefig(figure_path + "small_triangles_cbar_right.pdf")
    plt.show()

    plot_lockin_transposed(
        spectroscopy_xarray,
        colormap=colormap,
        figsize=(1.5, 1.25),
        loc_colorbar="right",
        y_axis_multiplier=1e-9,
        invert_current=invert_current_spectroscopy,
    )
    plt.savefig(figure_path + "small_spectroscopy_cbar_right.pdf")
    plt.show()

    print("g factor: ", get_g_factor(rabi_osc_state.info["magnetic_field"], 2.79e9))
    g_uncertainty = get_g_factor(
        rabi_osc_state.info["magnetic_field"] - mag_delta / 2, 2.79e9
    ) - get_g_factor(rabi_osc_state.info["magnetic_field"] + mag_delta / 2, 2.79e9)
    print("g uncertainty", g_uncertainty)
    plot_lockin_transposed(
        rabi_chevron_xarray,
        colormap=colormap,
        figsize=(1.5, 1.25),
        loc_colorbar="right",
        vline=rabi_osc_state.info["magnetic_field"],
        invert_current=invert_current_chevron,
    )
    plt.savefig(figure_path + "small_rabi_chevron_cbar_right.pdf")
    plt.show()

    plot_line(
        rabi_osc_state,
        figsize=(1.725, 1.625),
        invert_current=invert_current_chevron,
        sigma=sigma,
    )
    plt.savefig(figure_path + "small_rabi_osc.pdf")
    plt.show()


def get_guids(
    spectroscopy_state,
    rabi_chevron_state,
    bias_triangle_state,
    rabi_osc_state,
    ):
    guids = []
    local_ids = []
    guids.append(spectroscopy_state.data_identifiers["edsr_measurement"])
    guids.append(rabi_chevron_state.data_identifiers["rabi_measurement"])
    guids.append(rabi_chevron_state.info["pulsed_msmt_id"])
    guids.append(bias_triangle_state.data_identifiers["low_magnet_measurement"])
    guids.append(bias_triangle_state.data_identifiers["high_magnet_measurement"])
    guids.append(rabi_osc_state.data_identifiers["rabi_measurement"])

    local_ids.append(load_by_guid(spectroscopy_state.data_identifiers["edsr_measurement"]).captured_run_id)
    local_ids.append(load_by_guid(rabi_chevron_state.data_identifiers["rabi_measurement"]).captured_run_id)
    local_ids.append(load_by_guid(rabi_chevron_state.info["pulsed_msmt_id"]).captured_run_id)
    local_ids.append(load_by_guid(bias_triangle_state.data_identifiers["low_magnet_measurement"]).captured_run_id)
    local_ids.append(load_by_guid(bias_triangle_state.data_identifiers["high_magnet_measurement"]).captured_run_id)
    for guid in rabi_osc_state.data_identifiers["rabi_measurement"]:
        local_ids.append(load_by_guid(guid).captured_run_id)
    return guids, local_ids