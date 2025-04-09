import openslide
import os
from utils.h5 import load_hdf5
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2

# import matplotlib.pyplot as plt
# import seaborn as sns

# 设置cuda,全局变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from skimage.color import label2rgb
h5_path = r"/public/home/dapengtao_nj/zhy/project/mywsi/heatmap/mysvs_data-GINE-Transformer-kaiming-asl-adam-acd-text-batch8-nope"
img_path = r"/public/home/dapengtao_nj/zhy/project/mywsi/data/mysvs"
superpixel_path = (
    r"/public/home/dapengtao_nj/zhy/project/mywsi/graphs/mysvs/superpixels"
)
# save_path = r"/public/home/dapengtao_nj/zhy/project/mywsi/heatmap/mysvs/save4"
save_path = r"/public/home/dapengtao_nj/zhy/project/mywsi/heatmap/mysvs_data-GINE-Transformer-kaiming-asl-adam-acd-text-batch8-nope"

h5_path_list = os.listdir(h5_path)

# class_name = [
#     "G3",
#     "G4",
#     "G5",
#     "Normal",
#     "Stroma",
#     "Whole mount images",
#     "Biopsy images scanned",
# ]

# class_name = [
#     "Lymph node metastases",
#     "Tumour Type_Invasive ductal carcinoma",
#     "Tumour Type_Invasive lobular carcinoma",
#     "Tumour Type_Other type",
#     "ER_Positive",
#     "PR_Positive",
#     "HER2_Positive",
#     "Molecular subtype_HER2(+)",
#     "Molecular subtype_Luminal A",
#     "Molecular subtype_Luminal B",
#     "Molecular subtype_Triple negative",
#     "ALN status_N+(1-2)",
#     "ALN status_N+(>2)",
#     "ALN status_N0",
#     "T stage_T1",
#     "T stage_T2",
# ]

class_name = [
    "Lepidic",
    "In Situ",
    "Papillary",
    "Acinar",
    "Solid",
    "Micropapillary",
    "Cribriform",
    "Invasive",
    "Minimally Invasive",
]
for i, h5_file in enumerate(h5_path_list):
    print(f"正在处理第{i+1}/{len(h5_path_list)}: {h5_file}")
    # try:
    heatmap_result = load_hdf5(os.path.join(h5_path, h5_file))
    file_name = h5_file.split(".")[0]
    if os.path.exists(os.path.join(save_path, file_name)):
        print(f"{file_name} 已经处理过")
        continue
    # 处理heatmap
    feature_map = torch.from_numpy(heatmap_result["feature_map"])
    image_linear_weight = torch.from_numpy(heatmap_result["image_linear_weight"])
    text_linear_output = torch.from_numpy(heatmap_result["text_linear_output"])
    label = torch.from_numpy(heatmap_result["label"])
    # if label.sum() < 3:
    #     print(f"{file_name} 只有{label.sum()}个label")
    #     continue
    # out = torch.sigmoid(torch.from_numpy(heatmap_result["out"]))
    # # out保留两位小数
    # # out=torch.round(out * 100) / 100
    # # 判断out预测的和label是否一致，如果不一致break
    # # out大于0.5的位置置为1，否则置为0
    # out[out > 0.5] = 1
    # out[out <= 0.5] = 0
    # if (out != label).sum() != 0:
    #     print(out)
    #     print(label)
    #     print(f"{file_name} out和label不一致")
    #     continue
    w = text_linear_output @ image_linear_weight
    # map_list_sigmoid = torch.zeros(
    #     (feature_map.shape[0], w.shape[0])  # , dtype=torch.float16
    # )
    map_list_norm = torch.zeros(
        (feature_map.shape[0], w.shape[0])
    )  # , dtype=torch.float16
    for i in range(w.shape[0]):
        v = torch.sum(feature_map * w[i], axis=1)
        # v = torch.mean(feature_map * w[i], dim=1)
        # v_sigmoid = torch.sigmoid(v)
        # v_sigmoid小于0.5的位置置为0
        # v_sigmoid[v_sigmoid < 0.5] = 0
        # v = F.relu(v)
        v_norm = (v - v.min()) / (v.max() - v.min())
        # map_list_sigmoid[:, i] = v_sigmoid
        map_list_norm[:, i] = v_norm
    # 处理superpixel
    superpixel_file = os.path.join(superpixel_path, file_name + "_superpixel.npy")
    superpixel = torch.from_numpy(np.load(superpixel_file)).to(device)
    superpixel_size = superpixel.shape
    downsample_size = (superpixel_size[0] // 8, superpixel_size[1] // 8)
    # 加载图片
    img_file = os.path.join(img_path, file_name + ".svs")
    slide = openslide.OpenSlide(img_file)
    img = np.asarray(
        slide.get_thumbnail((superpixel_size[1], superpixel_size[0])).convert("RGB")
    )
    # img_file = os.path.join(img_path, file_name + ".jpg")
    # img = cv2.imread(img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(
    #     img, (superpixel_size[1], superpixel_size[0]), cv2.COLORMAP_JET
    # )
    idx = map_list_norm.nonzero(as_tuple=True)
    c_idx = label.nonzero(as_tuple=True)[1]
    # 对superpixel 赋值
    for c in c_idx:
        # 处理每个类别
        print(f"正在处理类别{class_name[c]}")
        idx_i = idx[1] == c
        seg_map = torch.zeros(
            (superpixel_size[0], superpixel_size[1])  # , dtype=torch.float16
        ).to(device)
        for i in tqdm(idx[0][idx_i]):
            mask = superpixel == (i + 1)
            seg_map[mask] = map_list_norm[i, c]
        save = os.path.join(save_path, file_name)
        os.makedirs(save, exist_ok=True)
        # 保存热力图
        heatmap = cv2.applyColorMap(
            np.uint8(255 * seg_map.cpu().numpy()), cv2.COLORMAP_JET
        )
        heatmap_down = cv2.resize(
            heatmap,
            (downsample_size[1], downsample_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(os.path.join(save, class_name[c] + "_heatmap.png"), heatmap_down)
        # 保存热力图+原图
        add_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        add_img_down = cv2.resize(
            add_img,
            (downsample_size[1], downsample_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(os.path.join(save, class_name[c] + "_add.png"), add_img_down)
        # 保存原图
    img_down = cv2.resize(
        img, (downsample_size[1], downsample_size[0]), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(os.path.join(save, "wsi.png"), img_down)
    del seg_map, superpixel
    torch.cuda.empty_cache()
    # except Exception as e:
    #     print(f"处理{h5_file}失败: {e}")
    #     continue
    # plt.figure(dpi=300)
    # plt.imshow(seg_map.cpu().numpy(), cmap="jet")
    # plt.axis("off")
    # plt.savefig(os.path.join(save_path, file_name + "_" + class_name[c] + ".png"))
    # plt.clf()
    # # 对superpixel 赋值
    # seg_map=torch.zeros((map_list_sigmoid.shape[1],superpixel_size[0],superpixel_size[1]),dtype=torch.float16).cuda()
    # for i,c in tqdm(idx):
    #     mask=(superpixel==(i+1))
    #     seg_map[c,mask]=map_list_sigmoid[i,c]
