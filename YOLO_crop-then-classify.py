import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def select_bbox_model_file():
    """
    Opens a file dialog to select the YOLO detection (bounding box) model file.
    
    Returns:
        str: The path to the selected detection model file.
    """
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select YOLO Bounding Box Model File",
        filetypes=[("PyTorch Model files", "*.pt"), ("All files", "*.*")]
    )
    if not model_path:
        print("No BBox model file selected. Exiting.")
        exit(1)
    return model_path

def select_cls_model_file():
    """
    Opens a file dialog to select the YOLO classification model file.
    
    Returns:
        str: The path to the selected classification model file.
    """
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select YOLO Classification Model File",
        filetypes=[("PyTorch Model files", "*.pt"), ("All files", "*.*")]
    )
    if not model_path:
        print("No Classification model file selected. Exiting.")
        exit(1)
    return model_path

def select_image_folder():
    """
    Opens a directory dialog to select the folder containing images.
    
    Returns:
        str: The path to the selected image folder.
    """
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    if not folder_path:
        print("No folder selected. Exiting.")
        exit(1)
    return folder_path

def detect_objects(frame, model):
    """
    Detects objects (bounding boxes) in the provided image using the YOLO detection model.
    
    Args:
        frame (np.ndarray): The input image.
        model (YOLO): The YOLO detection model.
    
    Returns:
        list: A list of bounding boxes, each represented as a numpy array [x1, y1, x2, y2].
    """
    results = model(source=frame,
                    half=True,         # Using half precision
                    retina_masks=False,
                    conf=0.5,          # Confidence threshold
                    iou=0.5,           # IoU threshold for NMS
                    imgsz=640)         # Input size
    bounding_boxes = []
    for r in results:
        for xyxy in r.boxes.xyxy:
            bounding_boxes.append(xyxy.cpu().numpy())
    return bounding_boxes

def get_top_class_from_image(model, image_input):
    """
    Runs the YOLO classification model on the given image input (can be a file path or np.ndarray)
    and returns the top predicted class label as provided by the model.
    
    Args:
        model (YOLO): The YOLO classification model.
        image_input: Input image data.
    
    Returns:
        str or None: The class label, or None if classification fails.
    """
    try:
        results = model(image_input)
        if not results:
            print("No results returned from classification.")
            return None
        result = results[0]
        # Try to use the 'pred' attribute first.
        if hasattr(result, "pred") and result.pred is not None:
            pred_val = result.pred
            if isinstance(pred_val, (list, tuple)):
                top_index = int(pred_val[0])
            elif hasattr(pred_val, "cpu"):
                top_index = int(pred_val.cpu().item())
            elif isinstance(pred_val, int):
                top_index = pred_val
            else:
                print(f"Unexpected type for pred: {type(pred_val)}")
                return None
            label = model.names.get(top_index, str(top_index))
            return label
        # Otherwise, fall back to using the 'probs' attribute.
        elif hasattr(result, "probs"):
            top_index = result.probs.top1
            label = model.names.get(top_index, str(top_index))
            return label
        else:
            print("No valid classification result found.")
            return None
    except Exception as e:
        print(f"Error processing classification: {e}")
        return None

def update_tag_file(image_path, tags):
    """
    Writes the unique classification tags to a text file corresponding to the image.
    The text file will have the same base name as the image with a '.txt' extension.
    
    Args:
        image_path (str): The full path to the image file.
        tags (set): A set of predicted class tags.
    """
    base, _ = os.path.splitext(image_path)
    txt_file = base + '.txt'
    tag_str = ','.join(sorted(tags))
    try:
        with open(txt_file, 'w') as f:
            f.write(tag_str)
    except Exception as e:
        print(f"Error updating tag file for '{image_path}': {e}")

def process_images(detection_model, cls_model, folder_path):
    """
    Processes each image in the dataset folder:
      1. Loads the image.
      2. Detects bounding boxes using the detection model.
      3. For each bounding box, crops the image and classifies the crop.
      4. Aggregates unique classification tags from all detected objects.
      5. Writes the aggregated tags to a corresponding text file.
    
    Args:
        detection_model (YOLO): The YOLO detection model.
        cls_model (YOLO): The YOLO classification model.
        folder_path (str): The path to the folder containing images.
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in allowed_extensions:
            continue
        print(f"Processing image: {file_name}")
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            continue

        # Detect bounding boxes.
        bboxes = detect_objects(image, detection_model)
        if not bboxes:
            print(f"No bounding boxes detected for image: {file_name}, skipping.")
            continue

        predicted_tags = set()
        # Process each detected bounding box.
        for bbox in bboxes:
            # bbox is [x1, y1, x2, y2] as a numpy array.
            x1, y1, x2, y2 = bbox.astype(int)
            # Ensure coordinates are within image bounds.
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            tag = get_top_class_from_image(cls_model, crop)
            if tag:
                predicted_tags.add(tag)
        if predicted_tags:
            print(f"Found tags for {file_name}: {predicted_tags}")
            update_tag_file(file_path, predicted_tags)
        else:
            print(f"No classification results for image: {file_name}")

def main():
    """
    Main pipeline function:
      1. Prompts the user to select the YOLO detection model file.
      2. Prompts the user to select the YOLO classification model file.
      3. Prompts the user to select the image folder (dataset).
      4. Loads the models and processes the images.
    """
    bbox_model_file = select_bbox_model_file()
    cls_model_file = select_cls_model_file()
    folder_path = select_image_folder()
    detection_model = YOLO(bbox_model_file)
    cls_model = YOLO(cls_model_file)
    process_images(detection_model, cls_model, folder_path)
    print("Processing complete.")

if __name__ == "__main__":
    main()
