from roboflow import Roboflow

rf = Roboflow(api_key="8YCSVwyfQ5xWsCgxq4pb")
project = rf.workspace("mars-vitij").project("mars-yrjkm")
dataset = project.version(1).download("yolov8", location="data/raw/mars-dataset")

print("Dataset downloaded!")
print(f"Location: data/raw/mars-dataset")