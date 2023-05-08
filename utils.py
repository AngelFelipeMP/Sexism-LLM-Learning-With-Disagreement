import os
import shutil
import pandas as pd
import gdown
import config
from zipfile import ZipFile
import json
from exist2023evaluation import main

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
            df_label.rename(columns={"soft_label": 'soft_label' + '_' + task }, inplace=True)
            
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


def merge_gold_soft_label(label_gold_path, dataset):
    for task in ['task1', 'task2', 'task3']:
        json_list = []
        for partition in ['training','dev']:
            
            path_label = label_gold_path + '/' + dataset + '_' + partition + '_' + task + '_gold_soft.json'
            with open(path_label, 'r') as file:
                data = json.load(file)
            json_list.append(data)

        path_merge = label_gold_path + '/' + dataset + '_' + 'training-dev' + '_' + task + '_gold_soft.json'
        json_list[0].update(json_list[1])
        with open(path_merge, 'w') as merged_file:
            json.dump(json_list[0], merged_file, indent=2)


def transformation(item):
    # dict to list ordered by keys
    if len(eval(item)) == 2:
        classes = [i[0] for i in sorted(eval(item).items())]
        values = [i[1] for i in sorted(eval(item).items())]
    else:
        classes = [i[0] for i in sorted(eval(item).items()) if i[0] != 'NO']
        values = [i[1] for i in sorted(eval(item).items()) if i[0] != 'NO']
    return classes, values


def save_preds(preds, df, task, data_train, data_val, epoch, transformer):
    index_label = [(index, eval(labels)) for index, labels in zip(df['id_EXIST'], df['soft_label_' + task])]
    classes, _ = transformation(str(index_label[0][1]))
    json_pred = {item[0]:{"soft_label":{}} for item in index_label}

    for i in range(len(index_label)):
        # for task 1
        if 'NO' not in index_label[0][1].keys():
            # add 'NO' value to json
            json_pred[index_label[i][0]]["soft_label"]['NO'] = index_label[i][1]['NO']
        # add other classes values to json
        for j, c in enumerate(classes):
            json_pred[index_label[i][0]]["soft_label"][c] = preds[i][j]
            
    # Save the dictionary as a JSON file
    path = config.LOGS_PATH + '/' + task + '_' + data_train + '_' + data_val + '_' + str(epoch) + '_' + transformer + '.json'
    with open(path, "w") as f:
        json.dump(json_pred, f, indent=2)


def eval_preds(task, data_train, data_val, epoch, transformer):
    path_json_results = config.LOGS_PATH + '/' + task + '_' + data_train + '_' + data_val + '_' + str(epoch) + '_' + transformer + '.json'
    
    if data_val == 'dev':
        path_gold_file = config.LABEL_GOLD_PATH + '/' + 'EXIST2023_' + data_val + '_' + task + '_gold_soft.json'
    else:
        path_gold_file = config.LABEL_GOLD_PATH + '/' + 'EXIST2023_' + data_train + '_' + task + '_gold_soft.json'
        
    result_icm_soft_soft = main(['-p', path_json_results, '-g', path_gold_file, '-t', task])
    
    return result_icm_soft_soft 
    



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