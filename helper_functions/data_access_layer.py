import json
import os
import datetime
import logging
import pickle
from typing import Any, Dict, Union

import numpy as np
from treelib import Node, Tree

from matplotlib import pyplot as plt

from helper_functions.readout_optimisation_visualisation import (
    plot_triangle_with_readout_box,
    box_params_to_real_space,
    results_to_real_space,
    plot_triangle_with_scores,
)
from pipelines.utils import timestamp_files

from bias_triangle_detection.coord_change import EuclideanTransformation


class DataAccess:
    def __init__(
        self, exp_name: str, root_folder: str = "../data/experiments/"
    ) -> None:
        self.exp_name = exp_name
        self.save_path_documentation = os.path.join(
            root_folder, exp_name, "documentation"
        )
        self.save_path_data = os.path.join(root_folder, exp_name, "extracted_data/")
        self.filename_tree = os.path.join(root_folder, exp_name, "node_states/tree")
        self.root_folder = root_folder
        try:
            self.tree = self.load_tree()
        except FileNotFoundError:
            self.tree = Tree()
        self.transform = None

    def check_file_exists(self, save_path: str) -> bool:
        return os.path.isfile(save_path)

    def get_current_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_with_figure(
        self,
        message: str,
        figure: Any,
        figure_name: str,
        markdown_name: str = "progression",
    ) -> None:
        """Create a figure with associated message.

        Args:
            message (str): Message to save.
            figure (Any): Figure object.
            figure_name (str): Name of the figure.
            markdown_name (str, optional): Markdown name. Defaults to 'progression'.
        """
        fig_file_name = timestamp_files() + "_" + figure_name + ".png"
        fig_path = os.path.join(
            self.root_folder, self.exp_name, "figures", fig_file_name
        )
        figure.savefig(fig_path)

        fig_path_in_md = os.path.join("..", "figures", fig_file_name).replace(
            "\\", "/"
        )  # Obsidian expects '/', not '\\'
        self.create_or_update_file(message, fig_path_in_md, markdown_name=markdown_name)

    def create_or_update_file(
        self,
        message: str,
        image_path: Union[str, None] = None,
        markdown_name: str = "progression",
    ) -> None:
        """Create or update a markdown file.

        Args:
            message (str): Message to save.
            image_path (str, optional): Path to image. Defaults to None.
            markdown_name (str, optional): Markdown name. Defaults to 'progression'.
        """
        logging.info(f"Saving message to markdown: {message}")
        save_path_documentation_md = os.path.join(
            self.save_path_documentation, markdown_name + ".md"
        )
        mode = "a" if self.check_file_exists(save_path_documentation_md) else "w"
        with open(save_path_documentation_md, mode) as md_file:
            md_file.write(f"\n## {self.get_current_timestamp()}\n")
            md_file.write(f"\n{message}\n")
            if image_path:
                md_file.write(f"\n![Image]({image_path})\n")

    def save_data(self, dict_with_data: Dict[str, Any]) -> None:
        """Save data to prespecified location.

        Args
            dict_with_data (dict): Dictionary with keys and the
                                   items to be saved
        """
        for key, item in dict_with_data.items():
            with open(self.save_path_data + key + ".pkl", "wb") as f:
                pickle.dump(item, f)

    def load_data(self, key: str) -> Any:
        with open(self.save_path_data + key + ".pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def save_tree(self) -> None:
        data = {
            node.identifier: [node.tag, node.data] for node in self.tree.all_nodes()
        }
        edges = [
            (
                self.tree.parent(n.identifier).identifier
                if self.tree.parent(n.identifier)
                else None,
                n.identifier,
            )
            for n in self.tree.all_nodes()
        ]
        root = self.tree.root
        with open(self.filename_tree, "w") as f:
            json.dump({"nodes": data, "edges": edges, "root": root}, f)

    # Load the tree from a file
    def load_tree(self) -> Tree:
        with open(self.filename_tree, "r") as f:
            data = json.load(f)
        tree = Tree()
        nodes = data["nodes"]
        edges = data["edges"]
        root = data["root"]
        tree.create_node(
            nodes[root][0], root, data=nodes[root][1]
        )  # Create the root node first
        nodes.pop(root)  # Remove the root from the nodes
        for node_id, (tag, data) in nodes.items():  # Create the rest of the nodes
            tree.create_node(tag, node_id, parent=root, data=data)
        for parent_id, node_id in edges:  # Establish the parent-child relationships
            if parent_id is not None and parent_id != root:
                tree.move_node(node_id, parent_id)
        return tree

    def set_readout_transform(
        self, pulsed_triangle_data: Any, box_data: Dict[str, Any]
    ) -> None:
        """
        Needed to plot the readout search stage.
        """
        self.pulsed_triangle_data = pulsed_triangle_data
        centroid: np.ndarray = box_data["centroid"]
        angle: float = box_data["angle"]
        angle = np.deg2rad(-angle)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        target_coords = ["tangent_param_0", "norm_param"]
        self.transform = EuclideanTransformation(
            rotation_matrix, centroid, ["V_RP", "V_LP"], target_coords
        )

    def plot_readout_box(self, pulsed_msmt: Any, box_data: Dict[str, Any]) -> None:
        """
        Needed to plot the readout search stage.
        """
        self.set_readout_transform(pulsed_msmt, box_data)
        centroid: np.ndarray = box_data["centroid"]
        w: float = box_data["w"]
        h: float = box_data["h"]
        fig = plot_triangle_with_readout_box(
            pulsed_msmt,
            center=centroid,
            box=box_params_to_real_space(h, w, self.transform),
        )
        message = "Box for readout defined"
        fig_name = "readout_box"
        self.create_with_figure(message, fig, fig_name)
        plt.close()

    def plot_readout_scores(self, results):
        """
        Needed to plot the readout search stage.
        """
        coords, scores = results_to_real_space(results, self.transform)
        fig = plot_triangle_with_scores(
            self.pulsed_triangle_data, coords=coords, scores=scores
        )
        message = "Scores of readout search"
        fig_name = "readout_search"
        self.create_with_figure(message, fig, fig_name)
        plt.close()


if __name__ == "__main__":
    # Usage
    da_layer = DataAccess("../Data/experiments/test/documentation/progression.md")
    da_layer.create_or_update_file(
        "This is a sample message",
        "../figures/test.png",
    )
