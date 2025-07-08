import glob
import os
import torch
import numpy as np 
from torchvision import models, transforms
from torchvision.io import read_image
import torch.nn as nn
import matplotlib.pyplot as plt

#TODO: pretrained parameter for resnet model is deprecated. switch to new alternative at some point

def load_model(model_path):
    device = torch.device("cpu")
    model = models.resnet18(pretrained =True)
    for param in model.parameters():
        param.requires_grad = False
    
    number_of_last_input = model.fc.in_features
    model.fc = nn.Linear(number_of_last_input,1)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    return model

def forward(image_tensor, model, correct):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = torch.sigmoid(model(image_tensor))

    predicted_class = ""

    if prediction.item() < 0.5:
        predicted_class = "C1"
    else:
        predicted_class = "C2"
    # show_image(image_tensor)
    return 1 if predicted_class == correct else 0


def forward_all(list, model, correct):
    results = np.zeros(len(list), dtype=np.uint8)

    for i, item in enumerate(list):
        results[i] = forward(item, model, correct)

    return results, np.count_nonzero(results==1)/len(list)

def show_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.squeeze(0).cpu() * std + mean
    img = img.clamp(0, 1)

    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = read_image(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def load_test_data(path):
    c1_path = os.path.join(path, "C1")
    c2_path = os.path.join(path, "C2")
    c1_files = glob.glob(f"{c1_path}/*")
    c2_files = glob.glob(f"{c2_path}/*")

    c1_tensors = []
    c2_tensors = []

    for item in c1_files:
        item = image_to_tensor(item)
        c1_tensors.append(item)

    for item in c2_files:
        item = image_to_tensor(item)
        c2_tensors.append(item)

    return c1_tensors, c2_tensors

if __name__ == "__main__":

    print("enter name of the dataset used:")
    model_name = input()
    model_path = os.path.join("models", f"{model_name}.pt")
    # C2_image_path = os.path.join("data/archive/test", "1.jpg")
    # C1_image_path = os.path.join("data/archive/test", "8.jpg")
    # C2_tensor = image_to_tensor(C2_image_path)
    # C1_tensor = image_to_tensor(C1_image_path)

    model = load_model(model_path)
    c1_test, c2_test = load_test_data("data/archive/test")

    _, c1_precicison = forward_all(c1_test, model, "C1") 
    _, c2_precicison = forward_all(c2_test, model, "C2") 

    print(f"C1 precicison = {c1_precicison}")
    print(f"C2 precicison = {c2_precicison}")