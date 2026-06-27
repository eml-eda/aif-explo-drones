# policy for experiment
AIF_POLICY = "wall_following"   # or "wall_following", "spiral", "pseudo_random", "rotate_and_measure", "aif_saccade", "aif_saccade_table"

# ---- YOLO detector model selection ----
# options:
# small: "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/yolo11_s_mixed_easy/mixed_detector_runs/yolo11s_ft_bottle_can_v1/weights/best.pt"
# nano:  "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/yolo11_n_mixed_easy/mixed_detector_runs/yolo11n_ft_bottle_can_v1/weights/best.pt"
# small (TABLE): "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/yolo11_s_mixed_tables/mixed_table_detector_runs/yolo11s_ft_bottle_can_table_v1/weights/best.pt"
AIF_YOLO_MODEL_SIZE = "small"       # options: "small", "nano"
AIF_YOLO_MODEL_VARIANT = "yolo11s" # options: "yolo11s", "yolo11n"
AIF_YOLO_MODEL = "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/yolo11_s_mixed_easy/mixed_detector_runs/yolo11s_ft_bottle_can_v1/weights/best.pt"

# ---- MobileNet detector model selection ----
# MobileNet variants
AIF_MOBILENET_MODEL_SIZE = "1.0"  # options example: "1.0", "0.75", "0.5"
AIF_MOBILENET_MODEL_VARIANT = "mbv2ssd_sigmoid_bottle_can_v1"
AIF_MOBILENET_MODEL = "/home/ale/dev/thesis_project/aif-explo-drones/detection_model_training/mobilenetv2_ssd_mixed_easy/runs/mbv2ssd_sigmoid_bottle_can_v1/best.pt"