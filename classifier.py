import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
import yaml
from easydict import EasyDict
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from CPGNet.models import MInterface
from CPGNet.dataloaders import DInterface
from utils.load_best_model import load_model_path_by_args
from utils.set_seeds import set_seed


def load_callbacks():
    callbacks = []
    callbacks.append(
        plc.EarlyStopping(monitor="avgval", mode="max", patience=10, min_delta=1)
    )
    callbacks.append(plc.TQDMProgressBar())
    callbacks.append(
        plc.ModelCheckpoint(
            monitor="avgval",
            filename="best-{epoch:02d}-{avgval:.3f}-{val_loss:.3f}",
            save_top_k=5,
            mode="max",
            save_last=True,
        )
    )

    if config.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def main(config, fold_num=0):
    config.fold_num = fold_num
    set_seed(config.seed)
    data_module = DInterface(**vars(config))
    model = MInterface(**vars(config))
    logger = TensorBoardLogger(
        save_dir=config.log_dir, name=config.log_name, version=config.log_v
    )
    callbacks = load_callbacks()
    trainer = Trainer(
        max_epochs=config.max_epochs,
        devices=config.cuda,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    if config.mode == "test":
        print("Testing...")
        load_paths = load_model_path_by_args(config)
        result_list = []
        for load_path in load_paths:
            result = trainer.test(model, data_module, ckpt_path=load_path)[0]
            result["model_ckpt"] = load_path.split("/")[-1]
            result["task"] = config.log_v
            result["dataset"] = config.dataset
            result_list.append(result)
        result_list.sort(key=lambda x: x["avg"], reverse=True)
        results = pd.DataFrame(result_list)
        results.to_excel(f"results/{config.dataset}-{config.log_v}.xlsx", index=False)
    elif config.mode == "train":
        trainer.fit(model, data_module)
        print("Training finished!")
        print("Testing...")
        load_paths = load_model_path_by_args(config)
        result_list = []
        for load_path in load_paths:
            result = trainer.test(model, data_module, ckpt_path=load_path)[0]
            result["model_ckpt"] = load_path.split("/")[-1]
            result["task"] = config.log_v
            result["dataset"] = config.dataset
            result_list.append(result)
        result_list.sort(key=lambda x: x["avg"], reverse=True)
        results = pd.DataFrame(result_list)
        results.to_excel(f"results/{config.dataset}-{config.log_v}.xlsx", index=False)
    elif config.mode == "predict":
        print("Predicting...")
        load_paths = load_model_path_by_args(config)
        result_list = []
        for load_path in load_paths:
            trainer.predict(model, data_module, ckpt_path=load_path)


if __name__ == "__main__":
    with open("config/config_class.yaml", encoding="utf-8") as f:
        file = f.read()
    config = yaml.load(file, yaml.FullLoader)
    config = EasyDict(config)
    main(config)
