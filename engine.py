import torch
import torch.nn as nn
import config

def loss_fn(outputs, targets):
    if targets.shape[1] == config.UNITS['task3']:
        return nn.BCELoss()(outputs, targets)
    else:
        return nn.CrossEntropyLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    fin_targets = []
    fin_predictions = []
    no_val_list = []
    total_loss = 0
    
    for batch in data_loader:
        targets = batch["targets"]
        no_value = batch["no_value"]
        del batch["targets"]
        del batch["no_value"]

        if targets.shape[1] != config.UNITS['task1']:
            # remove non-sexist tweets input/targets/preds
            batch, targets, no_indices = remove_non_sexist(targets, no_value, batch)
                
        batch = {k:v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, targets.to(device))
        total_loss += loss.cpu().detach().numpy().tolist()

        if targets.shape[1] != config.UNITS['task1']:
            # add non-sexist tweet targets/preds
            outputs, targets = no_sexist_pred(outputs, targets, no_indices, no_value)
            # task2 -> Normalize targets between 0 and (1 - No value)
            # task3 -> Normalize each target value between 0 and (1 - No value)
            outputs, targets = normalize_outputs(outputs, targets, no_value)

        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
        no_val_list.extend(no_value.view(-1).cpu().detach().numpy().tolist())
        
        if torch.cuda.device_count() > 1 and device != 'cpu' and config.DEVICE == 'max':
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        
    return no_val_list, fin_predictions, fin_targets, total_loss/len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_predictions = []
    no_val_list = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            targets = batch["targets"]
            no_value = batch["no_value"]
            del batch["targets"]
            del batch["no_value"]
            
            if targets.shape[1] != config.UNITS['task1']:
                # remove non-sexist tweets input/targets/preds
                batch, targets, no_indices = remove_non_sexist(targets, no_value, batch)
                
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(batch)
            loss = loss_fn(outputs, targets.to(device))
            total_loss += loss.cpu().detach().numpy().tolist()
            
            if targets.shape[1] != config.UNITS['task1']:
                # add non-sexist tweet targets/preds
                outputs, targets = no_sexist_pred(outputs, targets, no_indices, no_value)
                # task2 -> Normalize targets between 0 and (1 - No value)
                # task3 -> Normalize each target value between 0 and (1 - No value)
                outputs, targets = normalize_outputs(outputs, targets, no_value)
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
            no_val_list.extend(no_value.view(-1).cpu().detach().numpy().tolist())
    
    return no_val_list, fin_predictions, fin_targets, total_loss/len(data_loader)



def test_fn(data_loader, model, device):
    model.eval()
    fin_predictions = []
    no_val_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            no_value = batch["no_value"].view(-1,1)
            del batch["no_value"]
            
            if no_value.any():
                # remove non-sexist tweets input/targets/preds
                batch, _, no_indices = remove_non_sexist(None, no_value, batch)
                
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(batch)
            
            if no_value.any():
                # add non-sexist tweet targets/preds
                outputs, _ = no_sexist_pred(outputs, None, no_indices, no_value)
                # task2 -> Normalize targets between 0 and (1 - No value)
                # task3 -> Normalize each target value between 0 and (1 - No value)
                outputs, _ = normalize_outputs(outputs, None, no_value)
            
            fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
            no_val_list.extend(no_value.view(-1).cpu().detach().numpy().tolist())
    
    return no_val_list, fin_predictions

#### NEW FUNCS ####
def remove_non_sexist(targets, no_value, batch):
    no_indices, _ = torch.where(no_value > 0.84)
    for k,v in batch.items():
        batch[k] = torch.tensor([value for count, value in enumerate(v.numpy().tolist()) if count not in no_indices], dtype=torch.long)
    if targets != None:
        targets = torch.tensor([t for count, t in enumerate(targets.numpy().tolist()) if count not in no_indices], dtype=torch.float)
    
    return batch, targets, no_indices

def no_sexist_pred(outputs, targets, no_indices, no_value):
    vec_list = []
    for vec in [outputs, targets]:
        if vec == None:
            vec_list.append(None)
            break
        
        new_vec = []
        vec = vec.cpu().detach().numpy().tolist()
        i=0
        for index in range(len(no_value)):
            if index in no_indices:
                new_vec.append([0]*len(vec[0]))
            else:
                new_vec.append(vec[i])
                i+=1
        vec_list.append(torch.tensor(new_vec, dtype=torch.float))
    return vec_list[0], vec_list[1]


def normalize_outputs(outputs, targets, no_value):
    vec_list = []
    for vec in [outputs, targets]:
        if vec == None:
            vec_list.append(None)
            break
        
        if vec.shape[1] == config.UNITS['task2']:
            vec_normalized = (1 - no_value) * vec / vec.sum(dim=1)[:, None]
            vec_normalized = torch.nan_to_num(vec_normalized, nan=0.0)
        else:
            vec_normalized = vec * (1 - no_value)
        
        vec_list.append(vec_normalized)
    return vec_list[0], vec_list[1]




