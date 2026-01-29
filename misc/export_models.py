# export_models.py
import yaml
from stages.stage_2_persondetection import PersonDetectionStage
from stages.stage_3_weapondetection import WeaponDetectionStage

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Build stages with your normal config
    stage2 = PersonDetectionStage(cfg["stage_2"])
    stage3 = WeaponDetectionStage(cfg["stage_3"])

    # Export each stage as its own ONNX
    # Adjust input_size if your models expect a different resolution
    stage2.export_onnx("stage2_person.onnx", input_size=(1,3,640,640), opset=17)
    stage3.export_onnx("stage3_weapon.onnx", input_size=(1,3,640,640), opset=17)

