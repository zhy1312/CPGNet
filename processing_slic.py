import os
import numpy as np
import pandas as pd
from utils.processing import processing
import time
import gc
import torch
import cv2
from norm.normlizer import Normalizers
from utils.extractor import DeepFeatureExtractor
from utils.graph_builder import RAGGraphBuilder
from utils.set_seeds import set_seed
import openslide
from dgl.data.utils import save_graphs

from easydict import EasyDict
import yaml

# Load configuration file
with open("config/config_pro.yaml", encoding="utf-8") as f:
    file = f.read()
    config = yaml.load(file, yaml.FullLoader)
config = EasyDict(config)

# Set global random number
set_seed(config.seed)
print(f"Set the global random number seed:{config.seed}")
# Handle GPU
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data
dataset_dir = os.path.join("data", config.dataset)
dataset_list = [os.path.join(dataset_dir, path) for path in os.listdir(dataset_dir)]

feature_path = os.path.join(config.save_dir, config.dataset, "features")
if not os.path.exists(feature_path):
    os.makedirs(feature_path)
superpixels_path = os.path.join(config.save_dir, config.dataset, "superpixels")
if not os.path.exists(superpixels_path):
    os.makedirs(superpixels_path)
label = pd.read_csv(config.label_dir)

# Image Processing
print(f"The {config.dataset} is currently being processed...")
datasets_len = len(dataset_list)
for i, image_path in enumerate(dataset_list):
    _, image_name = os.path.split(image_path)
    image_name, image_type = image_name.split(".")
    print(f"{i+1}/{datasets_len}:{image_name}", end="\t")

    feature_save_path = os.path.join(feature_path, image_name + ".bin")
    if os.path.exists(feature_save_path):
        print("Completed, skip.")
        continue
    start1 = time.time()
    if image_type == "svs":
        slide = openslide.OpenSlide(image_path)
        pro = processing(slide)
        # Skip the larger images
        for level, dimension in enumerate(slide.level_dimensions):
            width, height = dimension
            size = width * height * 4 * 4 / 1024**3
            if size < config.MAX_SIZE:
                break
        img = np.asarray(
            slide.read_region(
                location=(0, 0),
                level=level,
                size=dimension,
            ).convert("RGB")
        )
        print(level, dimension)
    elif image_type in ["tif", "tiff"]:
        slide = openslide.OpenSlide(image_path)
        pro = processing(slide)
        # Skip the larger images
        for level, dimension in enumerate(slide.level_dimensions):
            width, height = dimension
            size = width * height * 4 * 4 / 1024**3
            if size < config.MAX_SIZE:
                break
        img = np.asarray(slide.get_thumbnail(dimension))
        print(level, dimension)
    elif image_type in ["jpg", "png", "jpeg"]:
        pro = processing()
        try:
            img = np.asarray(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1])
        except:
            print("big!")
            continue
        print(img.shape)

    contours = pro.Imageotsu(
        img,
        superpixel_size=config.data_processing.superpixel_size,
        filter_params=config.data_processing.filter_params,
    )
    mask = pro.Createmask(
        img,
        contours["foreground_contours"],
    )
    if config.data_processing.normalizer is not None:
        norm = Normalizers(
            method=config.data_processing.normalizer, fit=config.data_processing.V
        )
        img = norm.transform(img)
    end1 = time.time()
    print(f"The preprocessing has been completed and it took {end1-start1:.2f}s")
    # - contours["hole_area"]
    if os.path.exists(os.path.join(superpixels_path, image_name + "_superpixel.npy")):
        superpixel = np.load(
            os.path.join(superpixels_path, image_name + "_superpixel.npy")
        )
    else:
        superpixel = pro.Superpixel(
            img=img,
            superpixel_size=config.data_processing.superpixel_size,
            area=contours["foreground_area"],
            mask=mask,
            compactness=config.data_processing.compactness,
            min_size_factor=config.data_processing.min_size_factor,
            max_size_factor=config.data_processing.max_size_factor,
        )
        np.save(
            os.path.join(superpixels_path, image_name + "_superpixel.npy"), superpixel
        )
    end2 = time.time()
    print(
        f"The superpixel segmentation has been completed and it took {end2-end1:.2f}s."
    )
    extractor = DeepFeatureExtractor(
        architecture=config.feature_extractor.model_name,
        patch_size=config.feature_extractor.patch_size,
        resize_size=config.feature_extractor.resize_size,
        fill_value=config.feature_extractor.fill_value,
        batch_size=config.feature_extractor.batch_size,
        verbose=True,
        device=device,
    )
    features = extractor.process(img, superpixel)
    end3 = time.time()
    print(
        f"Feature extraction has been completed and it took {end3-end2:.2f}s, shape:{features.shape}."
    )
    graph_builder = RAGGraphBuilder(device=device, add_loc_feats=True)
    graph = graph_builder.process(superpixel, features)
    save_graphs(
        filename=feature_save_path,
        g_list=graph,
    )
    end4 = time.time()
    print(
        f"The construction of the graph has been completed and it took {end4-end3:.2f}s, {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
    )
    print(f"The image information has been saved and it took {end4-start1:.2f}s.")
    torch.cuda.empty_cache()
    del img, contours, mask, superpixel, features, graph, slide
    gc.collect()
    print()
    print("=" * 100)
    print()
print("done!")
