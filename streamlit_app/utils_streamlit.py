import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm
def load_class_names():
    with open("data/food-101/meta/classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def load_model(model_path):
    import timm
    model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=101)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    class_names = load_class_names()
    return model, class_names

def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)
        predicted_label = class_names[top_idx.item()]
    return predicted_label, top_prob.item() * 100
