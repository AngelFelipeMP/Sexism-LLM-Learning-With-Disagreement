import config 
# from utils import download_exist_package, download_data, process_EXIST2022_data
# from utils import download_exist_package, merge_data_labels
from utils import download_exist_package, merge_data_labels,test_to_csv,merge_training_dev

if __name__ == "__main__":
    # download_exist_package(
        # config.PACKAGE_PATH,
        # config.DATA_URL)
    
    # merge_data_labels(
    #     config.PACKAGE_PATH, 
    #     config.LABEL_GOLD_PATH, 
    #     config.DATA_PATH, 
    #     config.DATA)
    
    # test_to_csv(
    #     config.PACKAGE_PATH, 
    #     config.DATA_PATH, 
    #     config.DATA)
    
    merge_training_dev(
        config.DATA_PATH, 
        config.DATA)
    
    # download_data(config.DATA_PATH,
    #                 config.DATA_URL
    # )
    
    # process_EXIST2022_data(config.DATA_PATH, 
    #                 config.DATASET_CLASSES, 
    #                 config.DATASET_INDEX
    # )

