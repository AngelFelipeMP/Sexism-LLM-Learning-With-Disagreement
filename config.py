import os
#Hyperparameters
EPOCHS = 10 
MAX_LEN = 100
DROPOUT = 0.3
LR = 3e-5 #5e-6, 1e-5, 3e-5, 5e-5
BATCH_SIZE = 20 #24
TRANSFORMERS = ['bert-base-multilingual-uncased','xlm-roberta-base']

N_ROWS=None #16
SEED = 17
CODE_PATH = os.getcwd()
REPO_PATH = '/'.join(CODE_PATH.split('/')[0:-1])
DATA_PATH = REPO_PATH + '/' + 'data'
PACKAGE_PATH = REPO_PATH + '/' + 'EXIST_2023_Dataset'
DATA_DEV_PATH = PACKAGE_PATH + '/' + 'dev/EXIST2023_dev' + '.json'
DATA_TRAIN_PATH = PACKAGE_PATH + '/' + 'training/EXIST2023_training' + '.json'
DATA_TEST_PATH = PACKAGE_PATH + '/' + 'test/EXIST2023_test_clean' + '.json'
LABEL_GOLD_PATH = PACKAGE_PATH + '/' + 'evaluation/golds'

TRANSLATION_PATH = REPO_PATH + '/' + 'translations'
TRANSLATION_ROUNDTRIPS = 2

DATA = 'EXIST2023'
DATA_URL = 'URL_SHARED_BY_ORGANIZERS_TO_DOWNLOAD_EXIST_2023_DATA'

LABELS = ['task1', 'task2','task3']
COLUMN_TEXT = 'tweet'
COLUMN_LABELS = 'soft_label_'
DATASET_INDEX = 'id_EXIST'
UNITS = {'task1': 2, 'task2': 3, 'task3': 5}

DATASET_TRAIN = 'EXIST2023_training.csv'
DATASET_DEV = 'EXIST2023_dev.csv'
DATASET_TEST = 'EXIST2023_test.csv'
DATASET_TRAIN_DEV = 'EXIST2023_training-dev.csv'

DEVICE = 'mps' # 'max' # None 'max' 'cpu' 'cuda:0' 'cuda:1' 'mps'
TRAIN_WORKERS = 10
VAL_WORKERS = 10
LOGS_PATH = REPO_PATH + '/' + 'logs'
