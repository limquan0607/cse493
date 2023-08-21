import torch

SMALL_DATASET_DIR = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/SmallDataSet"
)

SMALL_DATA_PRE_1024 = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/SmallDataSet/Preprocess_v3"
)

SMALL_DATA_PRE_512 = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/SmallDataSet/Preprocess_v2"
)

SMALL_DATA_PRE_224 = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/SmallDataSet/Preprocess_224"
)

SMALL_TRAIN_DIR = ""
SMALL_VAL_DIR = ""
SMALL_DATA_PRE = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/SmallDataSet/Preprocess_v1"
)

MAIN_DATASET_DIR = "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/Task06_Lung"
MAIN_DATASET_PRE224 = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/Task06_Lung/Preprocess224"
    )
MAIN_TRAIN_DIR = ""
MAIN_VAL_DIR = ""
MAIN_DATASET_PRE = (
    "/home/nguye/cse493/LungTumor-Segmentation-main/dataset/Task06_Lung/Preprocess"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "/home/nguye/cse493/segementation/dataset/result"
