from get_model_tokenizer import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import *
from torch.optim import Adam


BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device("mps") # for Nvidia put it as CUDA


training_data, testing_data = get_datasets()

model = MyModel().to(DEVICE)

criterion = nn.BCELoss()
optimizer = (Adam(model.parameters(), lr= LR))



# model.to(DEVICE)

def train_model():
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle= True)
    for epoch in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        for indx, data in enumerate(train_dataloader):
            input, label = data
        
            input.to(DEVICE)
            label.to(DEVICE)
            
            prediction = model(input['input_ids'].reshape(8,250), 
                               input['attention_mask'].reshape(8,250))
            

            batch_loss = criterion(prediction, label.float().unsqueeze(1))
            total_loss_train += batch_loss.item()
            acc = (prediction.argmax(dim=1) == label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        print(f"Epoch no. {str(epoch)} Loss: {str(total_loss_train)} Accuracy: {str(total_acc_train)}")
        break
            

    
    
    