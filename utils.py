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
        no_value = [i[1] for i in sorted(eval(item).items()) if i[0] == 'NO']
    else:
        classes = [i[0] for i in sorted(eval(item).items()) if i[0] != 'NO']
        values = [i[1] for i in sorted(eval(item).items()) if i[0] != 'NO']
        no_value = [i[1] for i in sorted(eval(item).items()) if i[0] == 'NO']
        
    return classes, values, no_value


def save_preds(no_values, preds, df, task, data_train, data_val, epoch, transformer):
    dev_path = config.DATA_PATH + '/' + config.DATA + '_' + 'dev.csv'
    classes, _, _ = transformation(pd.read_csv(dev_path, index_col=None).at[0,'soft_label_' + task])
    json_pred = {item:{"soft_label":{}} for item in df['id_EXIST'].tolist()}
    
    for index, pred, no_value in zip(df['id_EXIST'], preds, no_values):
        for p,c in zip(pred, classes):
            json_pred[index]["soft_label"][c] = p
        
        if task != 'task1':
            json_pred[index]["soft_label"]['NO'] = 1.0 if no_value > 0.84 else no_value
    
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



def merge_roundtrip_translations(data_path, translation_path, package_path, gold_path, num_translation_roundtrips):
    
    dataset = "training"
    #read trainin data in JSON format
    path = package_path + '/' + dataset + '/EXIST2023_' + dataset + '.json'
    df_partition = pd.read_json(path, orient='index')
   
    #read goldstandard    
    golds = pd.DataFrame()
    for task in  config.UNITS.keys():
      
      goldfile = gold_path + '/EXIST2023_' + dataset + '_' + task + '_gold_soft.json' 
      

      gold_partition = pd.read_json(goldfile, orient='index')
      column = "soft_labels_{}".format(task)
      
      golds[column] = gold_partition
      
    golds["id_EXIST"] = golds.index

    merged = df_partition.merge(golds, on="id_EXIST")
    
    #read translations:
    translations = pd.DataFrame()
    for i in range(0, num_translation_roundtrips+1):
        translation = pd.read_csv(translation_path + '/EXIST2023_' + dataset + '.txt.' + str(i), sep='\t', header=None)  
        translations[str(i)] = translation

    ids = merged['id_EXIST'].tolist()

    #generate new IDs for the translations
    ids_1 = merged['id_EXIST'].apply(lambda x: str(x) + '_1')
    merged_1 = merged.copy()
    merged_1['id_EXIST'] = ids_1
    merged_1['tweet'] = translations['1']

    ids_2 = merged['id_EXIST'].apply(lambda x: str(x) + '_2')
    merged_2 = merged.copy()
    merged_2['id_EXIST'] = ids_2
    merged_2['tweet'] = translations['2']
    
    #pd.set_option('display.max_columns', None)

    #print(merged[['id_EXIST','tweet']].head())
    #print(merged_1[['id_EXIST','tweet']].head())
    #print(merged_2[['id_EXIST','tweet']].head())

    #write: M0, M1, M2, M0_M1, M0_M1, M0_M1_M2
    
    #id_EXIST, lang,	tweet,	number_annotators,	<
    # annotators,	gender_annotators,	age_annotators,	
    # labels_task1,	labels_task2,	labels_task3,	split,	
    # soft_label_task1,	soft_label_task2,	soft_label_task3

    #M0
    path_csv = data_path + '/' + 'EXIST2023_' + dataset

    m0 = merged
    m0_path = path_csv + '_M0.csv'
    m0.to_csv(m0_path, index=False)

    m1 =merged_1
    m1_path = path_csv + '_M1.csv'
    m1.to_csv(m1_path, index=False)

    m2 = merged_2
    m2_path = path_csv + '_M2.csv'
    m2.to_csv(m2_path, index=False)

    m0_m1 = pd.concat([m0,m1], ignore_index=True)
    m0_m1_path = path_csv + '_M0_M1.csv'
    m0_m1.to_csv(m0_m1_path, index=False)

    m0_m2 = pd.concat([m0,m2], ignore_index=True)
    m0_m2_path = path_csv + '_M0_M2.csv'
    m0_m2.to_csv(m0_m2_path, index=False)
    
    m0_m1_m2 = pd.concat([m0,m1,m2], ignore_index=True)
    
    m0_m1_m2_path = path_csv + '_M0_M1_M2.csv'
    m0_m1_m2.to_csv(m0_m1_m2_path, index=False)
    
     
    #Create dev CSV using the different translations
    dataset = "dev"

    #read dev data in JSON format
    path = package_path + '/' + dataset + '/EXIST2023_' + dataset + '.json'
    df_partition = pd.read_json(path, orient='index')
    
    #Read translations for dev:
    translations_dev = pd.DataFrame()
    for i in range(0, num_translation_roundtrips+1):
        translation = pd.read_csv(translation_path + '/EXIST2023_' + dataset + '.txt.' + str(i), sep='\t', header=None)  
        translations_dev[str(i)] = translation

    print(translations_dev.head())
    print(translations_dev.shape)
    print(df_partition.shape)
    path_csv = data_path + '/' + 'EXIST2023_' + dataset
    #create versions with different translations:
    #M0
    m0_dev = df_partition.copy()
    m0_dev.to_csv(path_csv + '_M0.csv', index=False)
    #M1
    m1_dev = df_partition.copy()
    m1_dev['tweet'] = translations_dev['1'].tolist()
    m1_dev.to_csv(path_csv + '_M1.csv', index=False)
    #M2
    m2_dev = df_partition.copy()
    m2_dev['tweet'] = translations_dev['2'].tolist()
    m2_dev.to_csv(path_csv + '_M2.csv', index=False)
    
    