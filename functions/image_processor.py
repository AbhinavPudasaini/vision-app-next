# image_processor.py

from PIL import Image
import requests
from io import BytesIO
import numpy as np
from datetime import datetime
import uuid
import torch
import os

from .model_loader import get_clip_model, get_face_app

from google.cloud import vision
from pymongo import MongoClient

import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Abhinav/Downloads/vision-app.json"

# Load models
clip_model, clip_processor = get_clip_model()
face_app = get_face_app()
# vision_client = vision.ImageAnnotatorClient()

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["photo_manager"]
collection = db["images"]


def load_image(source):
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")
    return image


def get_clip_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()


def get_face_embeddings(image: Image.Image):
    faces = face_app.get(np.array(image))
    embeddings = []
    for face in faces:
        embeddings.append({
            "bbox": face.bbox.tolist(),
            "embedding": face.embedding.tolist()
        })
    return embeddings

import cv2
import numpy as np

def compute_color_histogram(image_path, bins=(32, 32, 32)):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

# def get_dino_embedding(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img_tensor = preprocess(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         features = model.forward_features(img_tensor)["x_norm_clstoken"]
#         features = torch.nn.functional.normalize(features, dim=-1)

#     return features.cpu().numpy().astype("float32").squeeze()





# def analyze_image_with_vision(image_path):
#     with open(image_path, "rb") as img_file:
#         content = img_file.read()

#     image = vision.Image(content=content)

#     label_response = vision_client.label_detection(image=image)
#     labels = [label.description for label in label_response.label_annotations]

#     text_response = vision_client.text_detection(image=image)
#     texts = text_response.text_annotations
#     ocr_text = texts[0].description if texts else ""

#     return labels, ocr_text


def process_and_store_image(image_path):
    image = load_image(image_path)
    clip_emb = get_clip_embedding(image)
    face_embs = get_face_embeddings(image)
    # Compute color histogram
    color_histogram = compute_color_histogram(image_path)
    # labels, ocr_text = analyze_image_with_vision(image_path)
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    rel_path = os.path.relpath(os.path.abspath(image_path), static_dir).replace("\\", "/")

    metadata = {
        "image_id": str(uuid.uuid4()),
        "source": rel_path,
        "timestamp": datetime.utcnow(),
        "clip_embedding": clip_emb.tolist(),
        "faces": face_embs,
        "color_histogram": color_histogram,
        "quality": "unknown"
    }

    result = collection.insert_one(metadata)
    print(f"[✅] Stored image metadata with ID: {result.inserted_id}")
    return metadata


# from google.cloud import vision
# from pymongo import MongoClient
# from PIL import Image
# import io
# import os

# # Setup Google Vision API client
# vision_client = vision.ImageAnnotatorClient()

# # Setup MongoDB client
# mongo_client = MongoClient("mongodb://localhost:27017/")
# db = mongo_client["your_database_name"]
# collection = db["your_collection_name"]

# def analyze_image_with_vision(image_path):
#     with open(image_path, "rb") as img_file:
#         content = img_file.read()

#     image = vision.Image(content=content)

#     # Label detection (objects/concepts)
#     label_response = vision_client.label_detection(image=image)
#     labels = [label.description for label in label_response.label_annotations]

#     # OCR detection (text)
#     text_response = vision_client.text_detection(image=image)
#     texts = text_response.text_annotations
#     ocr_text = texts[0].description if texts else ""

#     return labels, ocr_text

# def update_db_with_vision_info(image_path):
#     labels, ocr_text = analyze_image_with_vision(image_path)

#     # Update MongoDB document
#     result = collection.update_one(
#         {"source": image_path},
#         {"$set": {
#             "vision_labels": labels,
#             "ocr_text": ocr_text
#         }}
#     )

#     if result.matched_count:
#         print(f"✅ Updated: {image_path}")
#     else:
#         print(f"⚠️ Image not found in DB: {image_path}")

# # 🔁 Example usage for a folder of images
# def process_folder(folder_path):
#     for file in os.listdir(folder_path):
#         if file.lower().endswith((".jpg", ".jpeg", ".png")):
#             img_path = os.path.join(folder_path, file)
#             update_db_with_vision_info(img_path)

# # Example:
# # process_folder("path_to_your_image_folder")



      # "vision_labels": labels,
        # "ocr_text": ocr_text,




    # metadata = process_and_store_image(image_path)
    # print("Image metadata stored:", metadata)
    
    # To display the image
    # img = load_image(metadata['source'])
    # img.show()  # Opens with default viewer