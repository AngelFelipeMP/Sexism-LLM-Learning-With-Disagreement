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
    output = config.PACKAGE_PATH.split('/')[-1] + '.zip'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    
    # loading the temp.zip and creating a zip object
    path_zip = config.CODE_PATH + '/' + output
    with ZipFile(path_zip, 'r') as zObject:
        
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(config.PACKAGE_PATH)


def merge_data_labels():
    
    
        

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