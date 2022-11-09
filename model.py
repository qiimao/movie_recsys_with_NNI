"""
Run main.py to start.

This script is modified from PyTorch quickstart:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import nni
# import torch model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# import helper method
from recSys import RecSys
from dataset import RatingDataset

df = pd.read_csv('./Rating.csv')

"""
Prepare Dataset
"""
Label_User_ID = preprocessing.LabelEncoder()
Label_Movie_ID = preprocessing.LabelEncoder()

df.User_ID = Label_User_ID.fit_transform(df.User_ID.values)
df.Movie_ID = Label_Movie_ID.fit_transform(df.Movie_ID.values)
  
df_train, df_valid = train_test_split(df, 
                                      test_size=0.2, 
                                      random_state=42, 
                                      stratify=df.Rating.values)
  
train_dataset = RatingDataset(User_ID = df_train.User_ID.values,
                              Movie_ID = df_train.Movie_ID.values,
                              Rating = (df_train.Rating.values-1))
  
valid_dataset = RatingDataset(User_ID = df_valid.User_ID.values,
                              Movie_ID = df_valid.Movie_ID.values,
                              Rating = (df_valid.Rating.values-1))
  
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=512, 
                                            shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                            batch_size=512, 
                                            shuffle=False)

"""
Setup Model
"""

# Build model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

model = RecSys(num_users = len(Label_User_ID.classes_), 
               num_movies = len(Label_Movie_ID.classes_)).to(device)

# Get optimized hyperparameters
params = {
    'lr': 0.001
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# Training functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        output = model(features[:,0],features[:,1],labels[:])
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features[:,0],features[:,1],labels[:])
            test_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    nni.report_intermediate_result(correct)
    return correct

# Train the model
epochs = 5
for t in range(epochs):
    train(train_loader, model, loss_fn, optimizer)
    accuracy = test(valid_loader, model, loss_fn)
    nni.report_intermediate_result(accuracy)

nni.report_final_result(accuracy)