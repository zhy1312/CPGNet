import numpy as np
from skimage.segmentation import slic
import cv2
from PIL import Image
import os
from scipy.spatial.distance import cosine
import gc
import torch

class processing:
    def Superpixel(
        self,
        img: np.array,
        superpixel_size=None,
        area=None,
        nr_superpixels=None,
        mask=None,
        compactness=5,
        min_size_factor=0.5,
        max_size_factor=3,
    ):
        if superpixel_size is not None and area is not None:
            nr_superpixels = int(area / superpixel_size)
        elif nr_superpixels is not None:
            nr_superpixels = nr_superpixels
        elif area is None:
            nr_superpixels = int(img.size[0] * img.size[1] / superpixel_size)
        return slic(
            img,
            n_segments=nr_superpixels,
            compactness=compactness,
            mask=mask,
            start_label=1,
            min_size_factor=min_size_factor,
            max_size_factor=max_size_factor,
        )

    def Imageotsu(
        self,
        img: np.array,
        use_otsu=False,
        superpixel_size=256,
        filter_params={
            "sthresh": 20,
            "mthresh": 7,
            "sthresh_up": 255,
            "close": 4,
            "a_t": 10,
            "a_h": 1,
            "max_n_holes": 8,
        },
    ):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(
            img_hsv[:, :, 1], filter_params["mthresh"]
        )  # Apply median blurring
        if use_otsu:
            _, img_otsu = cv2.threshold(
                img_med,
                0,
                filter_params["sthresh_up"],
                cv2.THRESH_OTSU + cv2.THRESH_BINARY,
            )
        else:
            _, img_otsu = cv2.threshold(
                img_med,
                filter_params["sthresh"],
                filter_params["sthresh_up"],
                cv2.THRESH_BINARY,
            )
        if filter_params["close"] > 0:
            kernel = np.ones((filter_params["close"], filter_params["close"]), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(
            img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # Find contours
        # scale = self.slide.level_downsamples[level]
        # scaled_ref_patch_area = int(ref_patch_size**2 / (scale * scale))
        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * superpixel_size
        filter_params["a_h"] = filter_params["a_h"] * superpixel_size

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # Find contours
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            contours = self._filter_contours(
                contours, hierarchy, filter_params
            )  # Necessary for filtering out artifacts

        return contours

    def Createmask(
        self,
        img: np.array,
        foreground_contours=None,
        hole_contours=None,
        save_path=None,
        img_name=None,
    ):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if foreground_contours is not None:
            cv2.drawContours(mask, foreground_contours, -1, (255, 255, 255), -1)
        if hole_contours is not None:
            for hole in hole_contours:
                cv2.drawContours(mask, hole, -1, (0, 0, 0), -1)
        if save_path is not None:
            mask_path = os.path.join(save_path, "mask")
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            _mask = Image.fromarray(mask)
            _mask.save(os.path.join(mask_path, f"{img_name}_mask.png"))
            del _mask
        return mask

    def _filter_contours(self, contours, hierarchy, filter_params):
        """
        Filter contours by: area.
        """
        filtered = []
        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.asarray(hole_areas).sum()
            if a == 0:
                continue
            if tuple((filter_params["a_t"],)) < tuple((a,)):
                filtered.append(cont_idx)
                all_holes.append(holes)
        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []
        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(
                unfiltered_holes, key=cv2.contourArea, reverse=True
            )
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params["a_h"]:
                    filtered_holes.append(hole)
            if filtered_holes != []:
                hole_contours.append(filtered_holes)
        foreground_area = 0.0
        for foreground in foreground_contours:
            a = cv2.contourArea(foreground)
            foreground_area += a
        hole_area = 0.0
        for hole_contour in hole_contours:
            a = cv2.contourArea(hole)
            for hole in hole_contour:
                hole_area += a
        return {
            "foreground_contours": foreground_contours,
            "hole_contours": hole_contours,
            "foreground_area": foreground_area,
            "foreground_len": len(foreground_contours),
            "hole_area": hole_area,
            "hole_len": sum([len(hole_contour) for hole_contour in hole_contours]),
        }
    def add_edge(self, g, k=20):
        def random_walk_pe(g, k):
            A = g.adjacency_matrix(scipy_fmt="csr")  # adjacency matrix
            RW = torch.tensor(A / (A.sum(1) + 1e-30)).cuda()  # 1-step transition probability
            # Iterate for k steps
            PE = [RW.diagonal()]
            RW_power = RW
            for _ in range(k - 1):
                RW_power = RW_power @ RW
                PE.append(RW_power.diagonal())
            RPE = torch.stack(PE, dim=-1).float()
            del A, RW, PE, RW_power
            torch.cuda.empty_cache()
            gc.collect()
            return RPE.cpu()


        def random_walk_pe_cpu(g, k):
            A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
            RW = torch.tensor(A / (A.sum(1) + 1e-30))  # 1-step transition probability
            # Iterate for k steps
            PE = [RW.diagonal()]
            RW_power = RW
            for _ in range(k - 1):
                RW_power = RW_power @ RW
                PE.append(RW_power.diagonal())
            RPE = torch.stack(PE, dim=-1).float()
            del A, RW, PE, RW_power
            torch.cuda.empty_cache()
            gc.collect()
            return RPE

        start, end = g.edges()
        feat = g.ndata["feat"]
        centroid = g.ndata["centroid"]
        # Create edge types
        edge_sim = []
        edge_Dist = []
        for idx_a, idx_b in zip(start, end):
            corr = 1 - cosine(feat[idx_a], feat[idx_b])
            a = centroid[idx_a]
            b = centroid[idx_b]
            edge_sim.append([torch.tensor(corr, dtype=torch.float)])
            edge_Dist.append([torch.sqrt(torch.pow(a - b, 2).sum())])
        edge_sim = torch.tensor(edge_sim)
        edge_Dist = torch.tensor(edge_Dist)
        efeat = torch.cat([edge_sim, edge_Dist], dim=1)
        mean = torch.mean(efeat, axis=0)
        std = torch.std(efeat, axis=0)
        norm_efeat = (efeat - mean) / std
        g.edata.update({"feat": norm_efeat})
        try:
            g.ndata["PE"] = random_walk_pe(g, k)
        except:
            g.ndata["PE"] = random_walk_pe_cpu(g, k)
        return g
