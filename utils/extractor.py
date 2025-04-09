import copy
import math
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from histocartography.pipeline import PipelineStep


class FeatureExtractor(PipelineStep):
    """Base class for feature extraction"""

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined instance_map

        Args:
            input_image (np.array): Original RGB image.
            instance_map (np.array): Extracted instance_map.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self._extract_features(input_image, instance_map)

    @abstractmethod
    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """
        Extract features from the input_image for the defined structure.

        Args:
            input_image (np.array): Original RGB image.
            structure (np.array): Structure to extract features.

        Returns:
            torch.Tensor: Extracted features
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """
        Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "features")

    @staticmethod
    def _preprocess_architecture(architecture: str) -> str:
        """
        Preprocess the architecture string to avoid characters that are not allowed as paths.

        Args:
            architecture (str): Unprocessed architecture name.

        Returns:
            str: Architecture name to use for the save path.
        """
        if architecture.startswith("s3://mlflow"):
            processed_architecture = architecture[5:].split("/")
            if len(processed_architecture) == 5:
                _, experiment_id, run_id, _, metric = processed_architecture
                return f"MLflow({experiment_id},{run_id},{metric})"
            elif len(processed_architecture) == 4:
                _, experiment_id, _, name = processed_architecture
                return f"MLflow({experiment_id},{name})"
            else:
                return f"MLflow({','.join(processed_architecture)})"
        elif architecture.endswith(".pth"):
            return f"Local({architecture.replace('/', '_')})"
        else:
            return architecture

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """
        Downsample an input image with a given downsampling factor.

        Args:
            image (np.array): Input tensor.
            downsampling_factor (int): Factor to downsample.

        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """
        Upsample an input image to a speficied new height and width.

        Args:
            image (np.array): Input tensor.
            new_height (int): Target height.
            new_width (int): Target width.

        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image


class HandcraftedFeatureExtractor(FeatureExtractor):
    """Helper class to extract handcrafted features from instance maps"""

    @staticmethod
    def _color_features_per_channel(img_rgb_ch, img_rgb_sq_ch, mask_idx, mask_size):
        codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
        hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
        feats_ = list(hist / mask_size)
        color_mean = np.mean(codes)
        color_std = np.std(codes)
        color_median = np.median(codes)
        color_skewness = skew(codes)

        codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
        color_energy = np.mean(codes)

        feats_.append(color_mean)
        feats_.append(color_std)
        feats_.append(color_median)
        feats_.append(color_skewness)
        feats_.append(color_energy)
        return feats_

    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """
        Extract handcrafted features from the input_image in the defined instance_map regions.

        Args:
            input_image (np.array): Original RGB Image.
            instance_map (np.array): Extracted instance_map. Different regions have different int values,
                                     the background is defined to have value 0 and is ignored.

        Returns:
            torch.Tensor: Extracted shape, color and texture features:
                          Shape:   area, convex_area, eccentricity, equivalent_diameter, euler_number, extent, filled_area,
                                   major_axis_length, minor_axis_length, orientation, perimiter, solidity;
                          Color:   Per channel (RGB) histogram with 8 bins:
                                   mean, std, median, skewness, energy;
                          Texture: entropy, glcm_contrast, glcm_dissililarity, glcm_homogeneity, glcm_energy, glcm_ASM
                                   (glcm = grey-level co-occurance matrix);
        """
        node_feat = []

        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img_square = np.square(input_image)

        img_entropy = Entropy(img_gray, disk(3))

        # For each instance
        regions = regionprops(instance_map)

        # pre-extract centroids to compute crowdedness
        centroids = [r.centroid for r in regions]
        all_mean_crowdedness, all_std_crowdedness = self._compute_crowdedness(centroids)

        for region_id, region in enumerate(regions):
            sp_mask = np.array(instance_map == region["label"], np.uint8)
            sp_rgb = cv2.bitwise_and(input_image, input_image, mask=sp_mask)
            sp_gray = img_gray * sp_mask
            mask_size = np.sum(sp_mask)
            mask_idx = np.where(sp_mask != 0)

            # Compute using mask [12 features]
            area = region["area"]
            convex_area = region["convex_area"]
            eccentricity = region["eccentricity"]
            equivalent_diameter = region["equivalent_diameter"]
            euler_number = region["euler_number"]
            extent = region["extent"]
            filled_area = region["filled_area"]
            major_axis_length = region["major_axis_length"]
            minor_axis_length = region["minor_axis_length"]
            orientation = region["orientation"]
            perimeter = region["perimeter"]
            solidity = region["solidity"]
            convex_hull_perimeter = self._compute_convex_hull_perimeter(
                sp_mask, instance_map
            )
            roughness = convex_hull_perimeter / perimeter
            shape_factor = 4 * np.pi * area / convex_hull_perimeter**2
            ellipticity = minor_axis_length / major_axis_length
            roundness = (4 * np.pi * area) / (perimeter**2)

            feats_shape = [
                area,
                convex_area,
                eccentricity,
                equivalent_diameter,
                euler_number,
                extent,
                filled_area,
                major_axis_length,
                minor_axis_length,
                orientation,
                perimeter,
                solidity,
                roughness,
                shape_factor,
                ellipticity,
                roundness,
            ]

            # (rgb color space) [13 x 3 features]
            feats_r = self._color_features_per_channel(
                sp_rgb[:, :, 0], img_square[:, :, 0], mask_idx, mask_size
            )
            feats_g = self._color_features_per_channel(
                sp_rgb[:, :, 1], img_square[:, :, 1], mask_idx, mask_size
            )
            feats_b = self._color_features_per_channel(
                sp_rgb[:, :, 2], img_square[:, :, 2], mask_idx, mask_size
            )
            feats_color = [feats_r, feats_g, feats_b]
            feats_color = [item for sublist in feats_color for item in sublist]

            # Entropy (gray color space) [1 feature]
            entropy = cv2.mean(img_entropy, mask=sp_mask)[0]

            # GLCM texture features (gray color space) [5 features]
            glcm = greycomatrix(sp_gray, [1], [0])
            # Filter out the first row and column
            filt_glcm = glcm[1:, 1:, :, :]

            glcm_contrast = greycoprops(filt_glcm, prop="contrast")
            glcm_contrast = glcm_contrast[0, 0]
            glcm_dissimilarity = greycoprops(filt_glcm, prop="dissimilarity")
            glcm_dissimilarity = glcm_dissimilarity[0, 0]
            glcm_homogeneity = greycoprops(filt_glcm, prop="homogeneity")
            glcm_homogeneity = glcm_homogeneity[0, 0]
            glcm_energy = greycoprops(filt_glcm, prop="energy")
            glcm_energy = glcm_energy[0, 0]
            glcm_ASM = greycoprops(filt_glcm, prop="ASM")
            glcm_ASM = glcm_ASM[0, 0]
            glcm_dispersion = np.std(filt_glcm)
            glcm_entropy = np.mean(Entropy(np.squeeze(filt_glcm), disk(3)))

            feats_texture = [
                entropy,
                glcm_contrast,
                glcm_dissimilarity,
                glcm_homogeneity,
                glcm_energy,
                glcm_ASM,
                glcm_dispersion,
                glcm_entropy,
            ]

            feats_crowdedness = [
                all_mean_crowdedness[region_id],
                all_std_crowdedness[region_id],
            ]

            sp_feats = feats_shape + feats_color + feats_texture + feats_crowdedness

            features = np.hstack(sp_feats)
            node_feat.append(features)

        node_feat = np.vstack(node_feat)
        return torch.Tensor(node_feat)

    @staticmethod
    def _compute_crowdedness(centroids, k=10):
        n_centroids = len(centroids)
        if n_centroids < 3:
            mean_crow = np.array([[0]] * n_centroids)
            std_crow = np.array([[0]] * n_centroids)
            return mean_crow, std_crow
        if n_centroids < k:
            k = n_centroids - 2
        dist = euclidean_distances(centroids, centroids)
        idx = np.argpartition(dist, kth=k + 1, axis=-1)
        x = np.take_along_axis(dist, idx, axis=-1)[:, : k + 1]
        std_crowd = np.reshape(np.std(x, axis=1), newshape=(-1, 1))
        mean_crow = np.reshape(np.mean(x, axis=1), newshape=(-1, 1))
        return mean_crow, std_crowd

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    def _compute_convex_hull_perimeter(self, sp_mask, instance_map):
        """Compute the perimeter of the convex hull induced by the input mask."""
        y1, y2, x1, x2 = self.bounding_box(sp_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= instance_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= instance_map.shape[0] - 1 else y2
        nuclei_map = sp_mask[y1:y2, x1:x2]

        if cv2.__version__[0] == "3":
            _, contours, _ = cv2.findContours(
                nuclei_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        elif cv2.__version__[0] == "4":
            contours, _ = cv2.findContours(
                nuclei_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        hull = cv2.convexHull(contours[0])
        convex_hull_perimeter = cv2.arcLength(hull, True)

        return convex_hull_perimeter


class PatchFeatureExtractor:
    """Helper class to use a CNN to extract features from an image"""

    def __init__(self, architecture: str, device: torch.device) -> None:
        """
        Create a patch feature extracter of a given architecture and put it on GPU if available.

        Args:
            architecture (str): String of architecture. According to torchvision.models syntax.
            device (torch.device): Torch Device.
        """
        self.device = device
        self.architecture = architecture
        if architecture.startswith("s3://mlflow"):
            model = self._get_mlflow_model(url=architecture)
        elif architecture.endswith(".pth"):
            model = self._get_local_model(path=architecture)
        elif architecture in ["uni", "conch"]:
            model = self._get_timm_moderl(architecture)
        else:
            model = self._get_torchvision_model(architecture)
        if architecture == "uni":
            self.num_features = 1024
            self.model = model
        elif architecture == "conch":
            self.num_features = 512
            self.model = model
        else:
            self.num_features = self._get_num_features(model)
            self.model = self._remove_classifier(model)
            self.model = torch.nn.parallel.DataParallel(
                module=self.model, device_ids=[0]
            )
        self.model.eval()

    @staticmethod
    def _get_num_features(model: nn.Module) -> int:
        """
        Get the number of features of a given model.

        Args:
            model (nn.Module): A PyTorch model.

        Returns:
            int: Number of output features.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            return model.fc.in_features
        else:
            classifier = model.classifier[-1]
            if isinstance(classifier, nn.Sequential):
                classifier = classifier[-1]
            return classifier.in_features

    def _get_local_model(self, path: str) -> nn.Module:
        """
        Load a model from a local path.

        Args:
            path (str): Path to the model.

        Returns:
            nn.Module: A PyTorch model.
        """
        model = torch.load(path, map_location=self.device)
        return model

    def _get_timm_moderl(self, model_name: str) -> nn.Module:
        import timm

        if model_name == "uni":
            model = timm.create_model(
                "vit_large_patch16_224",
                init_values=1e-5,
                num_classes=0,
            )
            model.load_state_dict(
                torch.load(
                    "checkpoints/uni/pytorch_model.bin",
                    map_location=self.device,
                ),
                strict=True,
            )
        elif model_name == "conch":
            from conch.open_clip_custom import create_model_from_pretrained
            from functools import partial

            model, _ = create_model_from_pretrained(
                "conch_ViT-B-16", "checkpoints/conch/pytorch_model.bin"
            )
            model.forward = partial(
                model.encode_image, proj_contrast=False, normalize=False
            )

        model = model.to(self.device)
        model = torch.nn.parallel.DataParallel(module=model, device_ids=[0])
        return model

    def _get_mlflow_model(self, url: str) -> nn.Module:
        """
        Load a MLflow model from a given URL.

        Args:
            url (str): Model url.

        Returns:
            nn.Module: A PyTorch model.
        """
        import mlflow

        model = mlflow.pytorch.load_model(url, map_location=self.device)
        return model

    def _get_torchvision_model(self, architecture: str) -> nn.Module:
        """
        Returns a torchvision model from a given architecture string.

        Args:
            architecture (str): Torchvision model description.

        Returns:
            nn.Module: A pretrained pytorch model.
        """

        from torchvision.models import ResNet101_Weights, resnet101

        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        model = model.to(self.device)
        return model

    @staticmethod
    def _remove_classifier(model: nn.Module) -> nn.Module:
        """
        Returns the model without the classifier to get embeddings.

        Args:
            model (nn.Module): Classifiation model.

        Returns:
            nn.Module: Embedding model.
        """
        if hasattr(model, "model"):
            model = model.model
        if isinstance(model, torchvision.models.resnet.ResNet):
            model.fc = nn.Sequential()
        else:
            model.classifier[-1] = nn.Sequential()
        return model

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Computes the embedding of a normalized image input.

        Args:
            image (torch.Tensor): Normalized image input.

        Returns:
            torch.Tensor: Embedding of image.
        """
        patch = patch.to(self.device)
        with torch.no_grad():
            embeddings = self.model(patch).squeeze()
        del patch
        return embeddings


class InstanceMapPatchDataset(Dataset):
    """Helper class to use a give image and extracted instance maps as a dataset"""

    def __init__(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        patch_size: int,
        stride: Optional[int],
        resize_size: int = None,
        fill_value: Optional[int] = 255,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Create a dataset for a given image and extracted instance map with desired patches
        of (patch_size, patch_size, 3). If fill_value is not None, it fills up pixels outside the
        instance maps with this value (all channels).

        Args:
            image (np.ndarray): RGB input image.
            instance map (np.ndarray): Extracted instance map.
            patch_size (int): Desired size of patch.
            stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                               patches of size patch_size are provided to the network. Defaults to None.
            fill_value (Optional[int]): Value to fill outside the instance maps
                                        (None means do not fill).
            mean (list[float], optional): Channel-wise mean for image normalization.
            std (list[float], optional): Channel-wise std for image normalization.
            transform (Callable): Transform to apply. Defaults to None.
        """
        self.image = image
        self.instance_map = instance_map
        self.patch_size = patch_size
        self.stride = stride
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.image = np.pad(
            self.image,
            (
                (self.patch_size, self.patch_size),
                (self.patch_size, self.patch_size),
                (0, 0),
            ),
            mode="constant",
            constant_values=fill_value,
        )
        self.instance_map = np.pad(
            self.instance_map,
            ((self.patch_size, self.patch_size), (self.patch_size, self.patch_size)),
            mode="constant",
            constant_values=0,
        )
        self.patch_size_2 = int(self.patch_size // 2)
        self.threshold = int(self.patch_size * self.patch_size * 0.25)
        self.properties = regionprops(self.instance_map)
        self.warning_threshold = 0.6
        self.patch_coordinates = []
        self.patch_instance_index = []
        self.patch_overlap = []

        basic_transforms = [transforms.ToPILImage()]
        if self.resize_size is not None:
            basic_transforms.append(transforms.Resize(self.resize_size))
        if transform is not None:
            basic_transforms.append(transform)
        basic_transforms.append(transforms.ToTensor())
        if self.mean is not None and self.std is not None:
            basic_transforms.append(transforms.Normalize(self.mean, self.std))
        self.dataset_transform = transforms.Compose(basic_transforms)

        self._precompute()
        self._warning()

    def _add_patch(self, center_x: int, center_y: int, index: int) -> None:
        """
        Extract and include patch information.

        Args:
            center_x (int): Centroid x-coordinate of the patch.
            center_y (int): Centroid y-coordinate of the patch.
            index (int): Instance index to which the patch belongs.
        """
        mask = np.zeros_like(self.instance_mask)
        mask[
            center_y
            - self.patch_size_2
            - self.offset_y : center_y
            + self.patch_size_2
            - self.offset_y,
            center_x
            - self.patch_size_2
            - self.offset_x : center_x
            + self.patch_size_2
            - self.offset_x,
        ] = 1

        overlap = np.sum(mask * self.instance_mask)
        if overlap > self.threshold:
            loc = [center_x - self.patch_size_2, center_y - self.patch_size_2]
            self.patch_coordinates.append(loc)
            self.patch_instance_index.append(index)
            self.patch_count += 1
            self.patch_overlap.append(overlap)

    def _get_patch(self, loc: list) -> np.ndarray:
        """
        Extract patch from image.

        Args:
            loc (list): Top-left (x,y) coordinate of a patch.
        """
        min_x = loc[0]
        min_y = loc[1]
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size
        return self.image[min_y:max_y, min_x:max_x]

    def _precompute(self):
        """Precompute instance-wise patch information for all instances in the input image."""
        for index, region in enumerate(self.properties):
            self.patch_count = 0

            # Extract center
            center_y, center_x = region.centroid
            center_x = int(round(center_x))
            center_y = int(round(center_y))

            # Bounding box
            min_y, min_x, max_y, max_x = region.bbox

            # Instant mask
            self.instance_mask = self.instance_map[
                min_y - self.patch_size_2 : max_y + self.patch_size_2,
                min_x - self.patch_size_2 : max_x + self.patch_size_2,
            ]
            self.instance_mask = np.array(self.instance_mask == region.label, dtype=int)
            self.offset_x = min_x - self.patch_size_2
            self.offset_y = min_y - self.patch_size_2

            # Extract patch information (coordinates, index)
            # quadrant 1
            y_ = copy.deepcopy(center_y)
            while y_ >= min_y:
                x_ = copy.deepcopy(center_x)
                while x_ >= min_x:
                    self._add_patch(x_, y_, index)

                    # Include at least one patch centered at the centroid
                    if self.patch_count == 0:
                        loc = [x_ - self.patch_size_2, y_ - self.patch_size_2]
                        self.patch_coordinates.append(loc)
                        self.patch_instance_index.append(index)
                        self.patch_count += 1
                    x_ -= self.stride
                y_ -= self.stride

            # quadrant 4
            y_ = copy.deepcopy(center_y)
            while y_ >= min_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, index)
                    x_ += self.stride
                y_ -= self.stride

            # quadrant 2
            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x)
                while x_ >= min_x:
                    self._add_patch(x_, y_, index)
                    x_ -= self.stride
                y_ += self.stride

            # quadrant 3
            y_ = copy.deepcopy(center_y) + self.stride
            while y_ <= max_y:
                x_ = copy.deepcopy(center_x) + self.stride
                while x_ <= max_x:
                    self._add_patch(x_, y_, index)
                    x_ += self.stride
                y_ += self.stride

    def _warning(self):
        """Check patch coverage statistics to identify if provided patch size includes too much background."""
        self.patch_overlap = np.array(self.patch_overlap) / (
            self.patch_size * self.patch_size
        )
        print(f"Fill ratio:{1-np.mean(self.patch_overlap):.3f}")
        if np.mean(self.patch_overlap) < self.warning_threshold:
            warnings.warn("Provided patch size is large")
            warnings.warn("Suggestion: Reduce patch size to include relevant context.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Loads an image for a given patch index.

        Args:
            index (int): Patch index.

        Returns:
            Tuple[int, torch.Tensor]: instance_index, image as tensor.
        """
        patch = self._get_patch(self.patch_coordinates[index])
        patch = self.dataset_transform(patch)
        return self.patch_instance_index[index], patch

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.patch_coordinates)


class DeepFeatureExtractor(FeatureExtractor):
    """Helper class to extract deep features from instance maps"""

    def __init__(
        self,
        architecture: str,
        patch_size: int,
        resize_size: int = None,
        stride: int = None,
        downsample_factor: int = 1,
        normalizer: Optional[dict] = None,
        batch_size: int = 64,
        fill_value: int = 255,
        num_workers: int = 0,
        verbose: bool = False,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        self.architecture = self._preprocess_architecture(architecture)
        self.patch_size = patch_size
        self.resize_size = resize_size
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride
        self.downsample_factor = downsample_factor
        self.verbose = verbose
        if normalizer is not None:
            self.normalizer = normalizer.get("type", "unknown")
        else:
            self.normalizer = None
        self.device = device
        super().__init__(**kwargs)

        if normalizer is not None:
            self.normalizer_mean = normalizer.get("mean", [0, 0, 0])
            self.normalizer_std = normalizer.get("std", [1, 1, 1])
        else:
            self.normalizer_mean = [0.485, 0.456, 0.406]
            self.normalizer_std = [0.229, 0.224, 0.225]
        self.patch_feature_extractor = PatchFeatureExtractor(
            architecture, device=self.device
        )
        self.fill_value = fill_value
        self.batch_size = batch_size
        self.architecture_unprocessed = architecture
        self.num_workers = num_workers
        if self.num_workers in [0, 1]:
            torch.set_num_threads(1)

    def _collate_patches(self, batch):
        """Patch collate function"""
        instance_indices = [item[0] for item in batch]
        patches = [item[1] for item in batch]
        patches = torch.stack(patches)
        return instance_indices, patches

    def _extract_features(
        self,
        input_image: np.ndarray,
        instance_map: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Extract features for a given RGB image and its extracted instance_map.

        Args:
            input_image (np.ndarray): RGB input image.
            instance_map (np.ndarray): Extracted instance_map.
            transform (Callable): Transform to apply. Defaults to None.
        Returns:
            torch.Tensor: Extracted features of shape [nr_instances, nr_features]
        """
        if self.downsample_factor != 1:
            input_image = self._downsample(input_image, self.downsample_factor)
            instance_map = self._downsample(instance_map, self.downsample_factor)

        image_dataset = InstanceMapPatchDataset(
            image=input_image,
            instance_map=instance_map,
            resize_size=self.resize_size,
            patch_size=self.patch_size,
            stride=self.stride,
            fill_value=self.fill_value,
            mean=self.normalizer_mean,
            std=self.normalizer_std,
            transform=transform,
        )

        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_patches,
        )
        features = torch.empty(
            size=(
                len(image_dataset.properties),
                self.patch_feature_extractor.num_features,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        embeddings = dict()
        for instance_indices, patches in tqdm(
            image_loader, total=len(image_loader), disable=not self.verbose
        ):
            emb = self.patch_feature_extractor(patches)
            for j, key in enumerate(instance_indices):
                if key in embeddings:
                    embeddings[key][0] += emb[j]
                    embeddings[key][1] += 1
                else:
                    embeddings[key] = [emb[j], 1]

        for k, v in embeddings.items():
            features[k, :] = v[0] / v[1]
        del image_loader
        return features.cpu().detach()
