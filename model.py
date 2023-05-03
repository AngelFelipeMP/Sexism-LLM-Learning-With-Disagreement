import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransforomerModel(nn.Module):
    def __init__(self, transformer, drop_out, number_of_classes):
        super(TransforomerModel, self).__init__()
        self.number_of_classes = number_of_classes
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, iputs):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)
        output = self.classifier(drop)
        
        # it means task 3 -> Multi-label classification
        if iputs['targets'].shape[1] > 3:
            output = self.softmax(output)

        return output
    
    #COMMENT: check if the best way to generate LLM final vector is to average the last hidden state
    #COMMENT: how to have different learning rate for the LLM and for the classifier 
    #COMMENT: I guess I will have to brack the model into two pieces