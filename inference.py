import os
import cv2
import json
import random
import time
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# ===== 1. 注册数据集（确保与训练时一致） =====
register_coco_instances("my_coco_val", {}, 
    "/mnt/workspace/small_coco_train/datasets/coco3200/annotations/instances_val2017.json", 
    "/mnt/workspace/small_coco_train/datasets/coco3200/val2017")

# 读取类别名称
with open("/mnt/workspace/small_coco_train/datasets/coco3200/annotations/instances_val2017.json", "r") as f:
    coco_data = json.load(f)
class_names = [cat["name"] for cat in coco_data["categories"]]
MetadataCatalog.get("my_coco_val").thing_classes = class_names
metadata = MetadataCatalog.get("my_coco_val")

# ===== 2. 配置模型 =====
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
cfg.MODEL.WEIGHTS = "/mnt/workspace/small_coco_train/output_my_coco/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 置信度阈值
cfg.DATASETS.TEST = ("my_coco_val", )

predictor = DefaultPredictor(cfg)

# ===== 3. 推理并可视化 =====
print("正在推理...")
start = time.time()

image_parent_path = "/mnt/workspace/small_coco_train/datasets/example"
image_name = "lzy.jpg"
image_path = image_parent_path + '/' + image_name
im = cv2.imread(image_path)
outputs = predictor(im)


v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
end = time.time()
# 推理完成，耗时: 1.91 秒
print(f"推理完成，耗时: {end - start:.2f} 秒")

# 保存结果图
output_path = "/mnt/workspace/small_coco_train/datasets/example/output" + '/' + image_name
cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
print(f"预测完成，结果保存在: {output_path}")
