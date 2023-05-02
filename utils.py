import os
import shutil
import pandas as pd
import gdown
import config
from zipfile import ZipFile

def download_exist_package(package_path, url):
    #create a data folder
    if os.path.exists(package_path):
        shutil.rmtree(package_path)
    os.makedirs(package_path)
    
    #download the exist package zip file
    output = package_path.split('/')[-1] + '.zip'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    
    # loading the temp.zip and creating a zip object
    path_zip = config.CODE_PATH + '/' + output
    with ZipFile(path_zip, 'r') as zObject:
        
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(package_path)


def merge_data_labels(package_path, label_gold_path, data_path, dataset):
    #create a data folder
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    # partition dev/traninig
    for partition in ['dev', 'training']:
        path_partition = package_path + '/' + partition + '/EXIST2023_' + partition + '.json'
        df_partition = pd.read_json(path_partition, orient='index')
        
        # Task 1/2/3
        for task in ['task1', 'task2', 'task3']:
            path_label = label_gold_path + '/' + dataset + '_' + partition + '_' + task + '_gold_soft.json'
            df_label = pd.read_json(path_label, orient='index')
            df_label.rename(columns={"soft_label": task + '_' + 'soft_label'})
            
            df_partition = pd.concat([df_partition, df_label], axis=1)
            
        path_csv = data_path + '/' + dataset + '_' + partition + '.csv'
        df_partition.to_csv(path_csv, index=False)


def test_to_csv(package_path, data_path, dataset):
    partition = 'test'
    path = package_path + '/' + partition + '/EXIST2023_' + partition + '_clean' + '.json'
    df_partition = pd.read_json(path, orient='index')
    
    path_csv = data_path + '/' + dataset + '_' + partition + '.csv'
    df_partition.to_csv(path_csv, index=False)


def merge_training_dev(data_path, dataset):
    partition_list =[]
    partition_strings_list = [] 
    # partition dev/traninig
    for partition in ['training', 'dev']:
        path_partition = data_path + '/' + dataset + '_' + partition + '.csv'
        df_partition = pd.read_csv(path_partition)
        partition_list.append(df_partition)
        partition_strings_list.append(partition)
            
    df_partition = pd.concat(partition_list)
            
    path_csv = data_path + '/' + dataset + '_' + '-'.join(partition_strings_list) + '.csv'
    df_partition.to_csv(path_csv, index=False)


#######################
### below old funcs ###
#######################


def process_EXIST2022_data(data_path, labels_col, index_col):
    files = [f for f in os.listdir(data_path) if 'processed' not in f]
    
    for file in files:
        df = pd.read_csv(data_path + '/' + file)
        print(df)
        
        if 'train' in file or 'dev' in file:
            df.replace(labels_col, inplace=True)
            print(df.head())

        dataset_name =  file[:-4] + '_processed' + '.csv'
        variable = 'DATASET' + ['_TRAIN' if 'train' in file else '_DEV' if 'dev' in file else '_TEST'][0]
        pass_value_config(variable, '\'' + dataset_name + '\'')
        
        df.to_csv(data_path + '/' + dataset_name, index=False,  index_label=index_col)


def pass_value_config(variable, value):
    with open(config.CODE_PATH + '/' + 'config.py', 'r') as conf:
        content = conf.read()
        new = content.replace(variable + ' = ' + "''", variable + ' = ' +  value )
        
    with open(config.CODE_PATH + '/' + 'config.py', 'w') as conf_new:
        conf_new.write(new)


def map_labels(df, labels_col):
    for col, labels in labels_col.items():
        df.replace({col:{number: string for string, number in labels.items()}}, inplace=True)
    return df