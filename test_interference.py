import glob
import os
import torch
from torchvision import models, transforms
from torchvision.io import read_image
import torch.nn as nn
import matplotlib.pyplot as plt

#TODO: read test data per cathegory from folders and test inference

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

    if prediction.item() < 0.5:
        print(f"prediction: C1 correct: {correct}")
    else:
        print(f"prediction: C2 correct: {correct}")

    # show_image(image_tensor)

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

if __name__ == "__main__":

    print("enter name of the dataset used:")
    model_name = input()
    model_path = os.path.join("models", f"{model_name}.pt")
    C2_image_path = os.path.join("data/archive/test", "1.jpg")
    C1_image_path = os.path.join("data/archive/test", "8.jpg")
    C2_tensor = image_to_tensor(C2_image_path)
    C1_tensor = image_to_tensor(C1_image_path)

    model = load_model(model_path)

    forward(C1_tensor, model, "C1")
    forward(C2_tensor, model, "C2")