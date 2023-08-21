import json
import os
import shutil

SOURCE = "/home/nguye/project/LungTumor-Segmentation-main/dataset/Task06_Lung"
DES = "/home/nguye/project/LungTumor-Segmentation-main/dataset/SmallDataSet"

with open("/home/nguye/project/LungTumor-Segmentation-main/dataset/SmallDataSet/dataset.json", "r") as f:
    data = json.load(f)

for i in data['training']:
    print("Creating small dataset...")
    image, label = i["image"][2:], i["label"][2:]
    source_image = os.path.join(SOURCE, image)
    source_label = os.path.join(SOURCE, label)
    des_image = os.path.join(DES, image)
    des_label = os.path.join(DES, label)
    shutil.copyfile(source_image, des_image)
    shutil.copyfile(source_label, des_label)
print("Done")
f.close()