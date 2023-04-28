import config 
# from utils import download_exist_package, download_data, process_EXIST2022_data
from utils import download_exist_package

if __name__ == "__main__":
    download_exist_package(config.PACKAGE_PATH,
                    config.DATA_URL
    )
    
    
    # download_data(config.DATA_PATH,
    #                 config.DATA_URL
    # )
    
    # process_EXIST2022_data(config.DATA_PATH, 
    #                 config.DATASET_CLASSES, 
    #                 config.DATASET_INDEX
    # )

