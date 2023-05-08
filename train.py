import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
import argparse
from utils import save_preds, eval_preds

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def train(df_train, df_val, task, epochs, transformer, max_len, batch_size, lr, drop_out, df_results, training_data):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[config.COLUMN_TEXT].values,
        target=df_train[config.COLUMN_LABELS + task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )
    
    val_dataset = dataset.TransformerDataset(
        text=df_val[config.COLUMN_TEXT].values,
        target=df_val[config.COLUMN_LABELS + task].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    #COMMENT: I may make the number_of_classes simpler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, drop_out, number_of_classes=config.UNITS[task]) 
    model.to(device)
    
    #NOTE: I must check appropriate no_decay for LLaMA
    #NOTE: differte learning rates for LLaMA and classifier
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
    
    # training and evaluation loop
    for epoch in range(1, epochs+1):
        
        pred_train, _ , loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        save_preds(pred_train, df_train, task, training_data, 'training', epoch, transformer)
        icm_soft_train = eval_preds(task, training_data, 'training', epoch, transformer)
        
        pred_val, _ , loss_val = engine.eval_fn(val_data_loader, model, device)
        save_preds(pred_val, df_val, task, training_data, 'dev', epoch, transformer)
        icm_soft_val = eval_preds(task, training_data, 'dev', epoch, transformer)
        
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'icm_soft_train': icm_soft_train,
                            'loss_train':loss_train,
                            'icm_soft_val':icm_soft_val,
                            'loss_val':loss_val
                        }, index=[0]
        )
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} ICM-soft_training = {:.3f} loss_training = {:.3f} ICM-soft_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, icm_soft_train, loss_train, icm_soft_val, loss_val))

        # save models weights
        path_model_save = config.LOGS_PATH + '/model' + '_' + task + '_' + training_data + '_' + transformer.split("/")[-1] + '.pt'
        if training_data == 'training-dev' and  epoch == epochs:
            torch.save(model.state_dict(), path_model_save)
        else:
            if epoch == 1 or icm_soft_val > df_results['icm_soft_val'][:-1].max():
                torch.save(model.state_dict(), path_model_save)

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, help='Datasets to train the models')
    args = parser.parse_args()
    
    if args.training_data == 'training':
        datasets = config.DATASET_TRAIN
        
    elif args.training_data == 'training-dev':
        datasets = config.DATASET_TRAIN_DEV
    
    elif not args.training_data:
        print('Specifying --training_data is required')
        exit(1)
    
    else:
        print('Specifying --training_data training OR training-dev')
        exit(1)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    df_train = pd.read_csv(config.DATA_PATH + '/' + datasets, index_col=None).iloc[:config.N_ROWS]
    df_train['NO_value'] = df_train['soft_label_task1'].apply(lambda x: eval(x)['NO'])
    
    df_val = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, index_col=None).iloc[:config.N_ROWS]
    df_val['NO_value'] = df_val['soft_label_task1'].apply(lambda x: eval(x)['NO']) 

    df_results = pd.DataFrame(columns=['task',
                                    'epoch',
                                    'transformer',
                                    'max_len',
                                    'batch_size',
                                    'lr',
                                    'dropout',
                                    'icm_soft_train',
                                    'loss_train',
                                    'icm_soft_val',
                                    'loss_val'
        ]
    )
    
    for task in tqdm(config.LABELS, desc='TRAIN', position=0):
        df_train = df_train[(df_train['NO_value']<=0.5) | (task == 'task1')]
        df_val = df_val[(df_val['NO_value']<=0.5) | (task == 'task1')]
        
        tqdm.write(f'\nTask: {task} Data: {args.training_data} Transfomer: {config.TRANSFORMERS.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE} Dropout: {config.DROPOUT} lr: {config.LR}')
            
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
                            args.training_data
        )
            
        df_results.to_csv(config.LOGS_PATH + '/' + args.training_data + '_' + task + '.csv', index=False)