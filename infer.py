# infer.py

import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load calorie map
calorie_map = pd.read_csv("food_calorie_map.csv").set_index("food")["calories"].to_dict()

# Class list (from folder names)
class_names = sorted(os.listdir("data/food-101/images"))

# Define model
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=101)
model.load_state_dict(torch.load("models/debug_efficientnetv2_food101.pth", map_location=device))
model.eval()
model.to(device)

# Preprocess image
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Predict
def predict(image_path):
    image_tensor = preprocess(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_index = torch.argmax(probs).item()
        pred_class = class_names[pred_index]
        calories = calorie_map.get(pred_class, "Unknown")
    return pred_class, calories, probs[pred_index].item()

# 🔍 Example usage
if __name__ == "__main__":
    test_img = "test_image.jpg"  # put a test image in root folder
    pred, cal, prob = predict(test_img)
    print(f"🍱 Predicted: {pred} ({prob:.2f} confidence)")
    print(f"🔥 Estimated Calories: {cal} kcal")
