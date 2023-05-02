import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
import argparse

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def train(df_train, df_val, task, epochs, best_epoch, transformer, max_len, batch_size, lr, drop_out, df_results, data):
# def train(df_train, task, epochs, best_epoch, transformer, max_len, batch_size, lr, drop_out, language, df_results, data):
    
    train_dataset = dataset.TransformerDataset(
        # text=df_train[language].values,
        text=df_train['tweet'].values,
        target=df_train[task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )
    
    val_dataset = dataset.TransformerDataset(
        text=df_val['tweet'].values,
        target=df_val[task].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, drop_out, number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1)
    model.to(device)
    
    #NOTE: I must check appropriate no_decay for LLaMA
    #TODO: differte learning rates for LLaMA and classifier
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001,},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]

    num_train_steps = int(len(df_train) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    for epoch in range(1, best_epoch+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        #TODO: replace f1 score and accuracy for (ICM-soft)
        # f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
        # acc_train = metrics.accuracy_score(targ_train, pred_train)
        
        pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device)
        #TODO: replace f1 score and accuracy for (ICM-soft)
        # f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
        # acc_train = metrics.accuracy_score(targ_train, pred_train)
        
        
        
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            #TODO: replace f1 score and accuracy for (ICM-soft)
                            # 'accuracy_train':acc_train,
                            # 'f1-macro_train':f1_train,
                            'loss_train':loss_train,
                            #TODO: replace f1 score and accuracy for (ICM-soft)
                            # 'accuracy_val':acc_val,
                            # 'f1-macro_val':f1_val,
                            'loss_val':loss_val
                        }, index=[0]
        )
        
        #TODO: I must check it: why pd.concat?
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        # tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train, f1_val, acc_val, loss_val))
        tqdm.write("Epoch {}/{} ICM-soft_training = {:.3f} loss_training = {:.3f} ICM-soft_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, icm-soft_train, loss_train, icm-soft_val, loss_val))

    #TODO: save best model
    # add "language" as input
    torch.save(model.state_dict(), f'{config.LOGS_PATH}/{data}_task[{task}]_transformer[{transformer.split("/")[-1]}]_epoch[{epoch}]_maxlen[{max_len}]_batchsize[{batch_size}]_dropout[{drop_out}]_lr[{lr}]_language[{language}].model')

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data/Datasets to train the models')
    args = parser.parse_args()
    
    #TODO: check if I need "domain"
    #TODO: if I need I must fix them
    if args.data == 'training':
        datasets = [config.DATASET_TRAIN]
        domain = config.DOMAIN_TRAIN
    
    elif args.data == 'training-dev':
        #TODO I have already joing it, chnage things here
        datasets = [config.DATASET_TRAIN, config.DATASET_DEV]
        domain = config.DOMAIN_TRAIN_ALL_DATA
    
    elif not args.data:
        print('Specifying --data is required')
        exit(1)
    
    else:
        print('Specifying --data training or training-dev')
        exit(1)
        
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    dataset_list = []
    for data in datasets:
        #TODO: I must check index_col=None
        df = pd.read_csv(config.DATA_PATH + '/' + data, index_col=None)
        dataset_list.append(df)
    
    #TODO: I have already joing it, chnage things here
    dfx = pd.concat(dataset_list, axis=0, ignore_index=True).iloc[:config.N_ROWS]
    print('Dataset shape: ', dfx.shape)
    
    df_results = pd.DataFrame(columns=['task',
                                    'epoch',
                                    'transformer',
                                    'max_len',
                                    'batch_size',
                                    'lr',
                                    'dropout',
                                    #TODO: replace f1 score and accuracy for (ICM-soft)
                                    # 'accuracy_train',
                                    # 'f1-macro_train',
                                    'loss_train',
                                    #TODO: replace f1 score and accuracy for (ICM-soft)
                                    # 'accuracy_val',
                                    # 'f1-macro_val',
                                    'loss_val'
        ]
    )
    
    for task in tqdm(config.LABELS, desc='TRAIN', position=0):
        #TODO: There will be no dfx anymre 
        df_train = dfx.loc[dfx[task]>=0]
        
        #TODO: Load df_val
        tqdm.write(f'\nTask: {task} Data: {domain} Transfomer: {config.TRANSFORMERS.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE} Dropout: {config.DROPOUT} lr: {config.LR}')
            
        df_results = train(df_train,
                            df_val,
                            task,
                            config.EPOCHS,
                            config.TRANSFORMERS,
                            config.MAX_LEN,
                            config.BATCH_SIZE,
                            config.LR,
                            config.DROPOUT,
                            df_results,
                            domain
        )
            
        df_results.to_csv(config.LOGS_PATH + '/' + domain + '.csv', index=False)