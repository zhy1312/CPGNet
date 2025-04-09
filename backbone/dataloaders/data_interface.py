import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader


class DInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_workers = kwargs["num_workers"]
        self.dataset = kwargs["dataset"]
        self.batch_size = kwargs["batch_size"]
        self.load_data_module()

    def setup(self, stage=None):
        if stage == "fit":
            self.trainset = self.instancialize(train="train")
            self.valset = self.instancialize(train="val")
        if stage == "test":
            self.testset = self.instancialize(train="test")
        if stage == "predict":
            self.predictset = self.instancialize(train="predict")

    def train_dataloader(self):
        return GraphDataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.valset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.testset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return GraphDataLoader(
            self.predictset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def load_data_module(self):
        name = "dataset"
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        """Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
