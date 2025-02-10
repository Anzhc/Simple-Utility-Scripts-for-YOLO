# Simple-Utility-Scripts-for-YOLO
SUS is a repository of simple utility scripts and pipelines for YOLO models.
#### YOLO_classify  
Simple script that will ask you to select a YOLO-cls model that you want to run, and then folder. It will run over all images in target folder and either create, or append detected class to existing .txt file. Modify this behaviour as you see fit.  
#### YOLO_crop-then-classify  
Same as above, but will ask for 2 YOLOs, first is detection, or segmnetation, it will perform cropping of target images, said crop is then used for CLS YOLO.  
Can we used for something that requires more strict input content, i.e. Faces.
