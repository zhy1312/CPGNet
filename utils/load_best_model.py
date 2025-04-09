import os
from pathlib2 import Path


def load_model_path(
    root_dir=None, name=None, version=None, v_num=None, best=False, key="acc"
):
    def sort_by_loss(path):
        name = path.stem
        loss = float(name.split("-")[-1].split("=")[-1])
        return loss

    def sort_by_acc(path):
        name = path.stem
        loss = float(name.split("-")[-2].split("=")[-1])
        return loss

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split("-")[-3].split("=")[-1])
        return epoch

    def generate_root():
        return str(Path(root_dir, name, version, "checkpoints"))

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith("best")]
        if key == "acc":
            files.sort(key=sort_by_acc, reverse=True)
            res = str(files[0])
        elif key == "loss":
            files.sort(key=sort_by_loss, reverse=False)
            res = str(files[0])
        elif key == "all":
            files.sort(key=sort_by_epoch, reverse=False)
            res = [str(i) for i in files]
    else:
        res = str(Path(root) / "last.ckpt")
    return res


def load_model_path_by_args(args):
    return load_model_path(
        root_dir=args.log_dir,
        name=args.log_name,
        version=args.log_v,
        v_num=args.load_ver,
        best=args.load_best,
        key=args.best_key,
    )
