import inspect
import torch
import importlib
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from utils.evaluate import evaluation
from .optimizer.lookahead import Lookahead
from .loss.asl import *
from .loss.dbl import *

import os
from utils.h5 import save_hdf5

# from .loss.Focal_Loss import *


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.tra_outs = []
        self.tra_labels = []
        self.val_outs = []
        self.val_labels = []
        self.test_outs = []
        self.test_labels = []
        self.save_path = os.path.join(
            self.hparams.save_dir,
            self.hparams.dataset + "-" + self.hparams.log_v,
        )
        os.makedirs(self.save_path, exist_ok=True)

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)
        out_sig = F.sigmoid(out.detach())
        self.tra_outs.append(out_sig)
        self.tra_labels.append(label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)
        out_sig = F.sigmoid(out.detach())
        self.val_outs.append(out_sig)
        self.val_labels.append(label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        *img, label, filename = batch
        out = self(img)
        loss = self.loss_function(out, label)
        out_sig = F.sigmoid(out.detach())
        self.test_outs.append(out_sig)
        self.test_labels.append(label)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        *img, label, filename = batch
        out, heatmap_result = self(img)
        loss = self.loss_function(out, label)
        heatmap_result["label"] = label.cpu().numpy()
        heatmap_result["out"] = out.cpu().numpy()
        save_hdf5(
            os.path.join(self.save_path, filename[0] + ".h5"), heatmap_result, mode="w"
        )
        out_sig = F.sigmoid(out.detach())
        self.test_outs.append(out_sig)
        self.test_labels.append(label)

    def on_train_epoch_end(self):
        outs = torch.cat(self.tra_outs).cpu().numpy()
        labels = torch.cat(self.tra_labels).cpu().numpy()
        self.tra_outs.clear()
        self.tra_labels.clear()
        eval = evaluation(outs, labels)
        self.log(
            "avgtra",
            eval[-1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        outs = torch.cat(self.val_outs).cpu().numpy()
        labels = torch.cat(self.val_labels).cpu().numpy()
        self.val_outs.clear()
        self.val_labels.clear()
        eval = evaluation(outs, labels)
        self.log(
            "avgval",
            eval[-1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        outs = torch.cat(self.test_outs).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        self.test_outs.clear()
        self.test_labels.clear()

        eval = evaluation(outs, labels)
        e = {
            "macroAP": eval[0],
            "macroAUC": eval[1],
            "microAP": eval[2],
            "microAUC": eval[3],
            "avg": eval[-1],
        }
        self.log_dict(dictionary=e, on_step=False, on_epoch=True, prog_bar=True)

    def on_predict_epoch_end(self):
        outs = torch.cat(self.test_outs).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        self.test_outs.clear()
        self.test_labels.clear()
        # np.save(os.path.join(self.save_path, "outs.npy"), outs)
        # np.save(os.path.join(self.save_path, "labels.npy"), labels)

    def configure_optimizers(self):
        if self.hparams.weight_decay is not None:
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
        elif self.hparams.optimizer == "lookaheadadam":
            optimizer = torch.optim.RAdam(
                self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay
            )
            optimizer = Lookahead(optimizer)
        else:
            raise ValueError("Invalid optimizer type!")

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == "steplr":
                scheduler = lrs.StepLR(
                    optimizer,
                    step_size=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            elif self.hparams.lr_scheduler == "cosinelr":
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lr_decay_steps,
                    eta_min=self.hparams.lr_decay_min_lr,
                )
            elif self.hparams.lr_scheduler == "multisteplr":
                scheduler = lrs.MultiStepLR(
                    optimizer,
                    milestones=self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate,
                )
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()

        if loss == "bce":
            self.loss_function = nn.BCEWithLogitsLoss()
        if loss == "mls":
            self.loss_function = nn.MultiLabelSoftMarginLoss()
        if loss == "fl":  # Focal Loss
            if self.hparams.freq_file is None:
                self.hparams.freq_file = {
                    "class_freq": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "neg_class_freq": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                }
            self.loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func="inv",
                focal=dict(focal=True, balance_param=1, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0,
                freq_file=self.hparams.freq_file,
            )
        if loss == "asl":  # Asymmetric Loss
            self.loss_function = AsymmetricLossOptimized(
                gamma_neg=1, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True
            )

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            Model = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}!"
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
