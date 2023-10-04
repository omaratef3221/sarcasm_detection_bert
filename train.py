from get_model_tokenizer import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import *
from torch.optim import Adam
import math

BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-3
DEVICE = torch.device("mps") # for Nvidia put it as CUDA


training_data, validation_data = get_datasets()
bert, _ = get_model_tokenizer()

    
model = MyModel(bert).to(DEVICE)

for param in bert.parameters():
    param.requires_grad = False
    
    
criterion = nn.BCELoss()
optimizer = (Adam(model.parameters(), lr= LR))



# model.to(DEVICE)

def train_model():
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle= True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle= True)

    for epoch in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        total_acc_val = 0
        total_loss_val = 0
        ## Training
        for indx, data in enumerate(train_dataloader):  # Training
            input, label = data
        
            input.to(DEVICE)
            label.to(DEVICE)

            prediction = model(input['input_ids'].squeeze(1), 
                               input['attention_mask'].squeeze(1)).squeeze(1)
            
     
            batch_loss = criterion(prediction.squeeze(), label.float())

            total_loss_train += batch_loss.item()
            acc = (torch.round(prediction).float() == label.float()).sum().item()
            total_acc_train += acc
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # model.zero_grad()
            
        ## Validation
        with torch.no_grad():
            for indx, data in enumerate(validation_dataloader):
                input, label = data
                input.to(DEVICE)
                label.to(DEVICE)
                
                prediction = model(input['input_ids'].squeeze(1), 
                               input['attention_mask'].squeeze(1)).squeeze(1)
                
                batch_loss_val = criterion(prediction.squeeze(), label.float())
                total_loss_val += batch_loss_val.item()
                acc = (torch.round(prediction).float() == label.float()).sum().item()
                total_acc_val += acc
                
        print(f'''Epoch no. {epoch + 1} Train Loss: {total_loss_train/1000:.4f} Train Accuracy: {(total_acc_train/(training_data.__len__())*100):.4f} Validation Loss: {total_loss_val/1000:.4f} Validation Accuracy: {(total_acc_val/(validation_data.__len__())*100):.4f}''')
        print("="*50)

    
    
    