import pandas as pd
from torch.utils.data import Dataset, DataLoader
from get_model_tokenizer import get_model_tokenizer
from sklearn.model_selection import train_test_split
import torch 
import numpy as np

data_df = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
data_df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data_df["headline"], data_df["is_sarcastic"], test_size=0.3)
X_train = np.array(X_train, dtype=str)
X_test = np.array(X_test, dtype=str)
y_train = np.array(y_train)
y_test = np.array(y_test)

_, tokenizer = get_model_tokenizer()
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = [tokenizer(x, 
                            max_length = 250,
                            truncation = True,
                            padding = 'max_length',
                            return_tensors='pt').to("mps")
                  for x in X
                 ] 
        
        self.Y = torch.tensor(Y).to("mps")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    
def get_datasets():
    training_data = dataset(X_train, y_train)
    testing_data = dataset(X_test, y_test)
    return training_data, testing_data