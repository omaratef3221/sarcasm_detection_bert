from transformers import AutoTokenizer, AutoModel
from torch import nn 
import torch

DEVICE = torch.device("mps")

def get_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return model, tokenizer

class MyModel(nn.Module):
    def __init__(self):
        
        super(MyModel, self).__init__()
        
        self.bert, _ = get_model_tokenizer()
        self.dropout = nn.Dropout(0.25)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask, return_dict = False)[0][:,0]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_result = self.sigmoid(linear_output)
        
        return final_result
    