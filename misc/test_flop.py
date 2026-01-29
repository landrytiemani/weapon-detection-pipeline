# script: profile_models.py
import torch
from thop import profile
from ultralytics import YOLO
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

def profile_model(model, input_size):
    model.eval()
    dummy_input = torch.randn(input_size)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    return {
        "Params": f"{params / 1e6:.2f}M",
        "GFLOPs": f"{(macs * 2) / 1e9:.2f}G"  # MACs ×2 = FLOPs
    }

# ------------------------
# Profile SSD MobilenetV2
ssd_model = create_mobilenetv2_ssd_lite(21, is_test=True)
ssd_model.load("weights/person/ssd_mobilenetV2_lite/mb2-ssd-lite-mp-0_686.pth")
print("SSD:", profile_model(ssd_model, (1, 3, 300, 300)))

# Profile YOLOv8n
yolo_model = YOLO("weights/person/yolov8n/yolov8n.pt").model
print("YOLOv8n:", profile_model(yolo_model, (1, 3, 640, 640)))

# Profile YOLOv8-EfficientViT
eff_model = YOLO("weights/weapon/efficientvit_yolov8/yolov8_efficientViT_best_V1.pt").model
print("YOLOv8-EfficientViT:", profile_model(eff_model, (1, 3, 640, 640)))

# Profile RT-DETR
rtdetr_model = YOLO("weights/weapon/rt_detr/RTDETR_v1_middle_best.pt").model
print("RT-DETR:", profile_model(rtdetr_model, (1, 3, 640, 640)))
