import numpy as np
from bias_triangle_detection.coord_change import EuclideanTransformation
import seaborn
from matplotlib import pyplot as plt


def box_params_to_real_space(h, w, transform):
    target_coords = ["tangent_param_0", "norm_param"]

    v_points = []
    points = [
        {"norm_param": -h / 2, "tangent_param_0": -w / 2},
        {"norm_param": +h / 2, "tangent_param_0": -w / 2},
        {"norm_param": +h / 2, "tangent_param_0": +w / 2},
        {"norm_param": -h / 2, "tangent_param_0": +w / 2},
        {"norm_param": -h / 2, "tangent_param_0": -w / 2},
    ]
    for transformed_point in points:
        rectangle_point = {name: transformed_point.pop(name) for name in target_coords}
        voltage_point = transform.inverse(rectangle_point)
        v_points.append(voltage_point)
        # lowerleft = transform(np.array([lp_center, rp_center]))
        # print(voltage_point)
    return v_points


def results_to_real_space(results, transform):
    target_coords = ["tangent_param_0", "norm_param"]

    v_points = []
    scores = []
    for result in results:
        score = result.score
        scores.append(score)
        # print(result.config)
        rectangle_point = {name: result.config[name] for name in target_coords}
        voltage_point = transform.inverse(rectangle_point)
        v_points.append(voltage_point)
        # lowerleft = transform(np.array([lp_center, rp_center]))
        # print(voltage_point, score)
    return v_points, scores


def plot_triangle_with_readout_box(
    data, name="I_SD", center=None, box=None, colormap="icefire"
):
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
    try:
        plt.colorbar(label=name + " [" + data[name].unit + "]")
    except:
        pass

    plt.scatter(
        [center[1]], [center[0]], marker="x", color="white", s=100, label="center"
    )
    marks_x = []
    marks_y = []
    for mark in box:
        marks_x.append(mark["V_LP"])
        marks_y.append(mark["V_RP"])
    plt.plot(marks_x, marks_y, "--x", color="white", label="box")
    plt.grid(False)
    # plt.show()
    plt.legend()
    return fig


def plot_triangle_with_scores(data, name="I_SD", coords=None, scores=None):
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
        cmap="Greys",
        alpha=1,
    )
    try:
        plt.ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
        plt.xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")
    except:
        print("! Unable to set axis labels")

    marks_x = []
    marks_y = []
    for mark in coords:
        marks_x.append(mark["V_LP"])
        marks_y.append(mark["V_RP"])

    plt.scatter(
        marks_x, marks_y, marker="o", s=30, c=scores, edgecolors="white", cmap="gnuplot"
    )
    plt.grid(False)
    try:
        plt.colorbar(label="readout optim score")
    except:
        print("! Unable to set axis labels")

    # plt.show()
    return fig
