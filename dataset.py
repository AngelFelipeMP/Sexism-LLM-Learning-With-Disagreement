import torch
import torch.nn as nn
from transformers import AutoTokenizer
from utils import transformation
import config

class TransformerDataset:
    def __init__(self, text, target, max_len, transformer):
        self.text = text
        self.target = target
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(config.REPO_PATH + '/' + transformer)
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        # task2/task3 Split labels NO from the others
        _, targets, no_value = transformation(self.target[item])
        
        # task2 -> Normalize targets between 0 and 1
        # task3 -> Normalize each target value between 0 and 1 
        if len(targets) != config.UNITS['task1']:
            targets = targets_normalization(targets, no_value)
        
        inputs = {k:torch.tensor(v, dtype=torch.long) for k,v in inputs.items()}
        inputs['targets'] = torch.tensor(targets, dtype=torch.float)
        inputs['no_value'] = torch.tensor(no_value, dtype=torch.float)
        
        return inputs
    
    
class TransformerDataset_Test:
    def __init__(self, text, no_task1, max_len, transformer):
        self.text = text
        self.no_task1 = no_task1
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(config.REPO_PATH + '/' + transformer)
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        inputs = {k:torch.tensor(v, dtype=torch.long) for k,v in inputs.items()}
        inputs['no_value'] = False if self.no_task1 is None else torch.tensor(self.no_task1[item], dtype=torch.float)
        
        return inputs
    
#### NEW FUNCTION #####
def targets_normalization(targets, no_value):
    if len(targets) == config.UNITS['task2']:
        normalized_list = [t/sum(targets) if sum(targets) != 0 else 0 for t in targets]
    else:
        normalized_list = [t/(1-no_value[0]) if sum(targets) != 0 else 0 for t in targets]
    
    return normalized_list