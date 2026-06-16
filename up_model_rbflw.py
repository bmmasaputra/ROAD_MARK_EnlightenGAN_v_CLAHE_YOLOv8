from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
WORKSPACE = os.getenv("WORKSPACE")

rf = Roboflow(api_key=API_KEY)
workspace = rf.workspace("skripsiworkspace-45mfj")

workspace.deploy_model(
  model_type="yolov8",
  model_path="DETECT/detect_ori_notune_ordered/comparison/ori_yolov8l",
  project_ids=["roadmark_night_taiwan"],
  filename="weights/ori_best.pt",
  model_name="v8_ori_best"
)