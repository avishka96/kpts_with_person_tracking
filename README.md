# kpts_with_person_tracking
YOLOv7 pose estimation with SORT person tracking

## Weights
Inside weights folder, add the yolov7 pose model
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

## Environment
Create the inter_local conda environment using the following command
conda env create -f environment.yml

## Steps to run code
Open anaconda prompt
Activate the conda environment using the following command
conda activate inter_local
Move into the directory containing the scripts
Run the following command
python batch_run.py <Path to the directory containing videos>


