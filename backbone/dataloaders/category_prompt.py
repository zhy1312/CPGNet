from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
import os
from PIL import Image
from pathlib import Path
import pickle


def createprompt(dataset, text_file):
    model, preprocess = create_model_from_pretrained(
        model_cfg="conch_ViT-B-16",
        checkpoint_path="checkpoints/conch/pytorch_model.bin",
    )
    _ = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    tokenizer = get_tokenizer()
    if dataset == "YNLUAD":
        text = [
            "Whole slide images of the adherent subtype of the lung using H&E stain",
            "Whole slide images of lung isoforms in situ using H&E stain",
            "Whole slide images of the Papillary subtype of the lung using H&E stain",
            "Whole slide images of the Acinar subtype of the lung using H&E stain",
            "Whole slide images of the Solid subtype of the lung using H&E stain",
            "Whole slide images of the Micropapillary subtype of the lung using H&E stain",
            "Whole slide images of the Cribriform subtype of the lung using H&E stain",
            "whole slide images of invasive adenocarcinoma using H&E stain",
            "whole slide images of minimally invasive adenocarcinoma using H&E stain",
        ]
    elif dataset == "BCNB":
        text = [
            "The histopathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images reveal Invasive Ductal Carcinoma",
            "The histopathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images reveal Invasive Lobular Carcinoma",
            "The histopathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images reveal Other Type",
            "The whole slide images of the Early Breast Cancer Core-Needle Biopsy show positive results for Estrogen Receptor (ER)",
            "The whole slide images of the Early Breast Cancer Core-Needle Biopsy show positive results for Progesterone Receptor (ER)",
            "The whole slide images of the Early Breast Cancer Core-Needle Biopsy show positive results for Human Epidermal GrowthFactor Receptor 2 (HER2)",
            "The molecular subtype of the Early Breast Cancer Core-Needle Biopsy whole slide images is Luminal A",
            "The molecular subtype of the Early Breast Cancer Core-Needle Biopsy whole slide images is Luminal B",
            "The molecular subtype of the Early Breast Cancer Core-Needle Biopsy whole slide images is HER2(+)",
            "The molecular subtype of the Early Breast Cancer Core-Needle Biopsy whole slide images is Triple negative",
            "The ALN status of the whole slide images of the Early Breast Cancer Core-Needle Biopsy is N+(1-2)",
            "The ALN status of the whole slide images of the Early Breast Cancer Core-Needle Biopsy is N+(>2)",
            "The ALN status of the whole slide images of the Early Breast Cancer Core-Needle Biopsy is N0",
            "The pathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images was TNM stage T1",
            "The pathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images was TNM stage T2",
            "The pathological diagnosis of the Early Breast Cancer Core-Needle Biopsy whole slide images showed lymph node metastasis",
        ]
    elif dataset == "AGGC":
        text = [
            "Whole slide images of the prostate for Gleason Pattern 3 (G3) using H&E stain.",
            "Whole slide images of the prostate for Gleason Pattern 4 (G4) using H&E stain.",
            "Whole slide images of the prostate for Gleason Pattern 5 (G5) using H&E stain.",
            "Whole slide images of the prostate for Normal using H&E stain.",
            "Whole slide images of the prostate for Stroma using H&E stain.",
            "The image type for Whole slide images is Whole mount images",
            "The image type of Whole slide images is Biopsy images scanned",
        ]
    tokenized_prompts = tokenize(texts=text, tokenizer=tokenizer).to(device)
    text_embedings = model.encode_text(tokenized_prompts)
    with open(text_file, "wb") as f:
        pickle.dump(text_embedings.cpu().detach(), f)
