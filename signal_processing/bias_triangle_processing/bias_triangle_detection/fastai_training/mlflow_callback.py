from __future__ import annotations

import tempfile

import mlflow
from fastai.basics import join_path_file
from fastai.callback.core import Callback
from fastai.callback.progress import Recorder
import numbers

__all__ = ["FastAIMLFlowCallback"]


class FastAIMLFlowCallback(Callback):
    "Log losses, metrics, model weights, model architecture summary to mlflow"
    order = Recorder.order + 1

    def __init__(self, log_model_weights=True):
        self.log_model_weights = log_model_weights
        self.step = 0
        self.epoch_step = 0

    def before_fit(self):
        try:
            mlflow.log_param("n_epoch", str(self.learn.n_epoch))
            mlflow.log_param("model_class", str(type(self.learn.model)))
        except:
            print("Did not log all properties.")

        try:
            with tempfile.NamedTemporaryFile(mode="w") as f:
                with open(f.name, "w") as g:
                    g.write(repr(self.learn.model))
                mlflow.log_artifact(f.name, "model_summary.txt")
        except:
            print("Did not log model summary. Check if your model is PyTorch model.")

        if self.log_model_weights and not hasattr(self.learn, "save_model"):
            print(
                "Unable to log model to mlflow.\n",
                'Use "SaveModelCallback" to save model checkpoints that will be logged to MLFlow.',
            )

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            mlflow.log_metric("batch__smooth_loss", self.learn.smooth_loss, self.step)
            mlflow.log_metric("batch__loss", self.learn.loss, self.step)
            mlflow.log_metric("batch__train_iter", self.learn.train_iter, self.step)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    mlflow.log_metric(f"batch__opt.hypers.{k}", v, self.step)
        self.step += 1

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ["epoch", "time"] and isinstance(v, numbers.Number):
                mlflow.log_metric(f"epoch__{n}", value=v, step=self.epoch_step)
            if n == "time":
                mlflow.log_text(str(v), f"epoch__{n}")

        # log model weights
        if self.log_model_weights and hasattr(self.learn, "save_model"):
            if self.learn.save_model.every_epoch:
                _file = join_path_file(
                    f"{self.learn.save_model.fname}_{self.learn.save_model.epoch}",
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            else:
                _file = join_path_file(
                    self.learn.save_model.fname,
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            if _file.exists():
                mlflow.log_artifact(_file)
