import os
import datetime
import logging

from matplotlib import pyplot as plt

from pipelines.utils import timestamp_files


class MarkdownFile:
    def __init__(
        self, exp_name, root_folder="../data/experiments/", file_name="progression.md"
    ):
        self.exp_name = exp_name
        self.save_path = os.path.join(root_folder, exp_name, "documentation", file_name)
        self.root_folder = root_folder

    def check_file_exists(self):
        return os.path.isfile(self.save_path)

    def get_current_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_with_figure(self, message, figure, figure_name):
        fig_file_name = timestamp_files() + "_" + figure_name + ".png"
        fig_path = os.path.join(
            self.root_folder, self.exp_name, "figures", fig_file_name
        )
        figure.savefig(fig_path)

        fig_path_in_md = os.path.join("../figures", fig_file_name)
        self.create_or_update_file(message, fig_path_in_md)

    def create_or_update_file(self, message, image_path=None):
        logging.info(f"Saving message to markdown: {message}")
        mode = "a" if self.check_file_exists() else "w"
        with open(self.save_path, mode) as md_file:
            md_file.write(f"\n## {self.get_current_timestamp()}\n")
            md_file.write(f"\n{message}\n")
            if image_path:
                md_file.write(f"\n![Image]({image_path})\n")


if __name__ == "__main__":
    # Usage
    md_file = MarkdownFile("../Data/experiments/test/documentation/progression.md")
    md_file.create_or_update_file(
        "This is a sample message",
        "../figures/test.png",
    )
