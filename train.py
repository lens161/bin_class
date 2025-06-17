import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from tqdm import tqdm

def prepare(dataset_name):
    train_dir = f"data/{dataset_name}/train"
    val_dir = f"data/{dataset_name}/validate"

    # transformation setting to turn images into 224, 224 tensors and use mean and std rgb values from resnet dataset for pretrained CNN
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),                                
                                       torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    val_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    #create datasets from training data
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=512)
    print(f"num iterations in train loader = {len(train_loader)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=512)

    device = torch.device("mps") # TODO: make adaptive to system 

    model = models.resnet18(pretrained=True) #use pretrained resnet model

    for params in model.parameters():
        params.requires_grad_ = False # freeze paramters of pretrained model so they dont update. pretrained part of the model stays the same

    # add final layer to convert to binary output
    number_of_last_input = model.fc.in_features
    model.fc = nn.Linear(number_of_last_input,1)

    model = model.to(device)
    
    return val_loader, train_loader, device, model

def train(model, train_loader:torch.utils.data.DataLoader, device):
    loss_function = BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.fc.parameters())
    epochs = 10
    losses = np.zeros(shape=epochs)
    validation_losses = np.zeros(shape=epochs)
    stop_if_no_change = 3 # stop after 3 consecutive epochs without change in loss
    stop_threshold = 0.03 # stop when loss reaches 0.03 (converges to 0)
    stop_counter = 0

    for epoch in range(epochs):
        print(f"training epoch {epoch+1}/{epochs}")
        total_loss = train_epoch(model, optimiser, loss_function, train_loader, device)
        print(f"epoch {epoch} loss = {total_loss}")
        losses[epoch] = total_loss
        val_loss = validate(model, loss_function, val_loader, device)
        validation_losses[epoch] = val_loss

        if epoch != 0:
            lowest_loss = min(losses[:epoch])
            if val_loss <= lowest_loss:
                curr_model = model.state_dict()
            if val_loss > lowest_loss:
                stop_counter += 1
            if (stop_counter == stop_if_no_change) or (lowest_loss <= stop_threshold):
                print(f"stop training - no change in loss or loss converged to 0")
                return curr_model
    
    return model.state_dict()

def train_epoch(model, optimiser, loss_function, train_loader:torch.utils.data.DataLoader, device):
    total_loss = 0
    model.train()
    for i, (x, y) in tqdm(enumerate(train_loader), total= len(train_loader)):
        x = x.to(device)
        y = y.unsqueeze(1).float() # add dimension to ground truth tensor to match prediction output (shape(batch_size, 1))
        y = y.to(device)

        logits = model(x)
        loss = loss_function(logits, y)

        loss.backward()
        total_loss += loss/len(train_loader)

        optimiser.step()
        optimiser.zero_grad()
        del logits, loss, x, y
    return total_loss

def validate(model, loss_function, val_loader:torch.utils.data.DataLoader, device):
    with torch.no_grad():
        total_loss = 0
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.unsqueeze(1).float() # add dimension to ground truth tensor to match prediction output (shape(batch_size, 1))
            y = y.to(device)
            logits = model(x)
            loss = loss_function(logits, y)
            total_loss += loss/len(val_loader)
            del loss, logits, x, y
    return total_loss

if __name__ == "__main__":
    print("enter dataset name (make shure dataset is prepared (prepare_data.py)):")
    dataset_name = input()
    val_loader, train_loader, device, model = prepare(dataset_name)
    model_weights = train(model, train_loader, device)

    model_path = f"models/{dataset_name}.pt"
    torch.save(model_weights, model_path)

