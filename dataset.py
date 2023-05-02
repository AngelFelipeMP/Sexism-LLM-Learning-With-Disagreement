import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TransformerDataset:
    def __init__(self, text, target, max_len, transformer):
        self.text = text
        self.target = target
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        # text = str(self.text[item])
        # text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        inputs = {k:torch.tensor(v, dtype=torch.long) for k,v in inputs.items()}
        inputs['targets'] = torch.tensor(self.target[item], dtype=torch.long)
        #TODO: Convert string to dict/list and than to torch.long tensor
        
        return inputs
    
    
class TransformerDataset_Test:
    def __init__(self, text, max_len, transformer):
        self.text = text
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        # text = str(self.text[item])
        # text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        inputs = {k:torch.tensor(v, dtype=torch.long) for k,v in inputs.items()}
        
        return inputs