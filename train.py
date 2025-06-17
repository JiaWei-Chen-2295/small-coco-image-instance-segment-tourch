import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def main():

    # 1. 注册数据集（名字任意，后面训练和评估用）
    register_coco_instances("my_coco_train", {}, 
                            "/mnt/workspace/small_coco_train/datasets/coco3200/annotations/instances_train2017.json", 
                            "/mnt/workspace/small_coco_train/datasets/coco3200/train2017")
    register_coco_instances("my_coco_val", {}, 
                            "/mnt/workspace/small_coco_train/datasets/coco3200/annotations/instances_val2017.json", 
                            "/mnt/workspace/small_coco_train/datasets/coco3200/val2017")

    # 2. 配置
    cfg = get_cfg()

    # 使用官方预训练的Mask R-CNN模型配置和权重（可根据需要改）
    from detectron2 import model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


    # 3. 设置训练数据集和验证数据集
    cfg.DATASETS.TRAIN = ("my_coco_train",)
    cfg.DATASETS.TEST = ("my_coco_val",)

    # 4. 设置类别数，必须和json里的categories数相等
    import json

    with open("/mnt/workspace/small_coco_train/datasets/coco3200/annotations/instances_train2017.json", "r") as f:
        coco_data = json.load(f)
    categories = coco_data["categories"]
    class_names = [cat["name"] for cat in categories]
    MetadataCatalog.get("my_coco_train").thing_classes = class_names
    metadata = MetadataCatalog.get("my_coco_train")
    num_classes = len(metadata.thing_classes)
    print(f"Number of classes: {num_classes}")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # 5. 训练相关参数
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2           # batch size
    cfg.SOLVER.BASE_LR = 0.00025            # 学习率
    cfg.SOLVER.MAX_ITER = 1000               # 训练迭代次数，数据小可以少设
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # 每张图片训练多少样本，默认128
    cfg.OUTPUT_DIR = "./output_my_coco"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 6. 训练
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
