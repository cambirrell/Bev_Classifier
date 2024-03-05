import json
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import DataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
import torchvision
from PIL import Image
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from torchvision.transforms import Compose, Resize, Lambda, ToTensor
import warnings
# Suppress the specific UserWarning
warnings.filterwarnings("ignore", message="The default value of the antialias parameter.*", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))


# Define the train and test datasets
#train and test datasets are split across a few folders but contain test or train in their name
class BevDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/cnb42/compute/images', split='train', transform=None, subset=None, step_one=False, get_id=False):
        self.root = root
        self.transform = transform
        self.split = split
        self.subset = subset
        self.step_one = step_one
        self.get_id = get_id
        # #use a reg ex to find all the files in the root directory that contain the word train or test
        # self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if split in f]
        #if train is spesified, open all folders from train_0 to train_69 and add the images in their subdirectories to the files list
        if split == 'train':
            for i in range(70):
                self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root + '/train_' + str(i)) for f in filenames]
        #if test is spesified, open all folders from test_0 to test_14 and add the images in their subdirectories to the files list
        if split == 'test':
            for i in range(15):
                self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root + '/test_' + str(i)) for f in filenames]
        # print(len(self.files))
        #remove any files that are not .jpg
        self.files = [f for f in self.files if '.jpg' in f]
        

        #
        #if a subset is specified, then only add the subdirectory that contains the subset name
        if self.subset:
            cat = json.load(open('catagories.json'))
            valid = cat[self.subset]
            self.files = [f for f in self.files if os.path.basename(os.path.dirname(f)) in valid]
        #set the labels based on the subdirectory name
        self.labels = [os.path.basename(os.path.dirname(f)) for f in self.files]
        #if step_one is true, read the catagories .json file and set the labels to the key of the value where the current label is an element of the value
        
        if step_one:
            self.labels = [self.get_label(f) for f in self.labels]
        elif get_id:
            self.labels = [os.path.basename(os.path.dirname(f)) for f in self.files]
        else:
            self.cat = json.load(open('catagories.json'))
            self.labels = [self.get_label_step_two(f) for f in self.labels]
            
    def get_label(self, f):
            cat = json.load(open('catagories.json'))
            for k, v in cat.items():
                if f in v:
                    return k
    def get_label_step_two(self, f):
        valid = self.cat[self.subset]
        return valid.index(f)
        
    def __getitem__(self, idx):
        #open the image and apply the transform
        img = Image.open(self.files[idx])
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        #return the image and the label
        if self.get_id:
            return img, self.labels[idx]
        self.labels = [int(i) for i in self.labels]
        if self.subset and not self.step_one:
            cat = json.load(open('catagories.json'))
            valid = cat[self.subset]
            return img, F.one_hot(torch.tensor(self.labels[idx]), len(valid)).float() 
        return img, F.one_hot(torch.tensor(self.labels[idx]), 17).float() if self.step_one else self.labels[idx]
    
    def __len__(self):
        return len(self.files)


#loop through the train and test datasets and apply the transforms to make sure it all works
# Define the transforms
transform = transforms.Compose([
    transforms.Resize((224, 448)),  # Rescale to 224x448
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize colors
])
# print('train data loading')
# train_dataset = BevDataset(transform=transform, split='train', step_one=True)
# # test_dataset = BevDataset(transform=transform, split='test')
# data = DataLoader(train_dataset, batch_size=32, shuffle=True)
# for x, y in train_dataset:
#     print(x.shape, y.shape) 
     
    
#make a resnet that takes in 3 channel images then outputs 17 classes
class ResNetStepOne(nn.Module):
    def __init__(self):
        super(ResNetStepOne, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.load_state_dict(torch.load('resnet18_pretrained.pth'))
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, 17)
        
    def forward(self, x):
        return self.resnet(x)
    
#Step one training
def train_step_one(model, train_loader, val_loader, epochs=9, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    loop = tqdm(total=len(train_loader)*epochs, position=0, leave=False)
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            #make y a one hot vector
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            loop.update(1)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                predicted = torch.argmax(y_hat, 1)
                actual = torch.argmax(y, 1) 
                total += y.size(0)
                correct += (predicted == actual).sum().item()
        print(f'Epoch {epoch}, Validation Accuracy: {correct / total}')
    return model

# Step one training test
dataset = BevDataset(transform=transform, split='train', step_one=True)
train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
data = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = ResNetStepOne()
model = train_step_one(model, data, val_data)

#make 16 resnets that take in 3 channel images then the amount of output classes are the amount of element in the items for each key in the categories.json file
class ResNetStepTwo(nn.Module):
    def __init__(self, output_classes):
        super(ResNetStepTwo, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.load_state_dict(torch.load('resnet18_pretrained.pth'))
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, output_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
#Step two training - train the expert models
#train all 16 models at once by spreading the data across the gpus
def train_step_two(model, train_loader, val_loader, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.to(device)
    # loop = tqdm(total=len(train_loader)*epochs, position=0, leave=False)
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            #make y a one hot vector
            optimizer.zero_grad()
            y_hat = model(x)
            # print(y_hat.shape, y.shape)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            # loop.set_description(f"Epoch [{epoch}/{epochs}]")
            # loop.set_postfix(loss=loss.item())
            # loop.update(1)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                predicted = torch.argmax(y_hat, 1)
                actual = torch.argmax(y, 1) 
                total += y.size(0)
                correct += (predicted == actual).sum().item()
    print(f'Epoch {epoch}, Validation Accuracy: {correct / total}')
    return model

catagories = json.load(open('catagories.json'))
expert_models = {k: ResNetStepTwo(len(v)) for k, v in catagories.items()}
for k, v in expert_models.items():
    print("Catagory:", k)
    dataset = BevDataset(transform=transform, split='train', subset=k, step_one=False)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    data = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False)
    expert_models[k] = train_step_two(expert_models[k], data, val_data, epochs=10)


def pred_id(subset, indx):
    valid = catagories[subset]
    return valid[indx]
#evaluation, use the trained models to predict the labels of the test data
#run throught the step one model and then depending on the output, run through the corresponding step two model
#return top 1 and top 5 accuracy and put all the results in a dataframe and then save it to a txt file
def evaluate(expert_models, step_one_model, test_loader):
    step_one_model.eval()
    for k, v in expert_models.items():
        v.eval()
    top_1_correct = 0
    top_5_correct = 0
    total = 0
    image_names = []
    top_1_predictions = []
    top_5_predictions = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = step_one_model(x)
            predicted = torch.argmax(y_hat, 1)
            second_predicted = torch.argsort(y_hat, 1)[:, -5:]
            # expert_models[predicted.item()].to(device)
            y_hat_expert = expert_models[predicted.item()](x)
            # expert_models[predicted.item()].to('cpu')
            # expert_models[second_predicted.item()].to(device)
            y_hat_expert_second = expert_models[second_predicted.item()](x)
            y_hat_expert_joint_prob = y_hat_expert * torch.max(y_hat, 1)
            y_hat_expert_second_joint_prob = y_hat_expert_second * torch.sort(y_hat, 1)[0][:, -5:]
            # expert_models[second_predicted.item()].to('cpu')
            #take the top 1 and top 5 predictions from the expert models
            all_predictions = torch.cat((y_hat_expert_joint_prob, y_hat_expert_second_joint_prob), 1)
            top_1_pred = torch.argmax(all_predictions, 1)
            top_5_pred = torch.argsort(all_predictions, 1)[:, -5:]
            #get the actual labels
            top_1_id = pred_id(predicted.item() if top_1_pred < len(catagories[predicted.item()]) else second_predicted.item(), top_1_pred if top_1_pred < len(catagories[predicted.item()]) else top_1_pred - len(catagories[predicted.item()]))
            top_5_ids = [pred_id(predicted.item() if top_5_pred[i] < len(catagories[predicted.item()]) else second_predicted.item(), top_5_pred[i] if top_5_pred[i] < len(catagories[predicted.item()]) else top_5_pred[i] - len(catagories[predicted.item()])) for i in range(5)]
            top_1_predictions.append(top_1_id)
            top_5_predictions.append(top_5_ids)
            #add one to the total
            total += y.size(0)
            #add one to the top 1 correct if the top 1 prediction is correct
            top_1_correct += (top_1_id == y).sum().item()
            #add one to the top 5 correct if the actual label is in the top 5 predictions
            top_5_correct += (y in top_5_ids).sum().item()
        #put the results in a dataframe for top one and another for top 5
        top_1_results = pd.DataFrame({'Image': image_names, 'Actual': y, 'Top_1': top_1_predictions})
        top_5_results = pd.DataFrame({'Image': image_names, 'Actual': y, 'Top_5': top_5_predictions})
        #save the results to two txt files for top 1 and top 5
        top_1_results.to_csv('top_1_results.txt', sep='\t')
        top_5_results.to_csv('top_5_results.txt', sep='\t')
        #make one last txt file for the top 1 and top 5 accuracy
        with open('accuracy.txt', 'w') as f:
            f.write(f'Top 1 accuracy: {top_1_correct / total}, Top 5 accuracy: {top_5_correct / total}')

#save models  
torch.save(model.state_dict(), 'step_one_model.pth')
for k, v in expert_models.items():
    torch.save(v.state_dict(), f'{k}_model.pth')
