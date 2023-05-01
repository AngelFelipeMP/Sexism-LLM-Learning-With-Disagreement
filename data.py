import config 
import argparse
from utils import download_exist_package, merge_data_labels,test_to_csv,merge_training_dev

parser = argparse.ArgumentParser()
parser.add_argument("--download_exist_package", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--merge_data_labels", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--test_to_csv", default=False, help="Must be True or False", action='store_true')
parser.add_argument("--merge_training_dev", default=False, help="Must be True or False", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    
    if args.download_exist_package:
        download_exist_package(
            config.PACKAGE_PATH,
            config.DATA_URL)
    
    if args.merge_data_labels:
        merge_data_labels(
            config.PACKAGE_PATH, 
            config.LABEL_GOLD_PATH, 
            config.DATA_PATH, 
            config.DATA)
    
    if args.test_to_csv:
        test_to_csv(
            config.PACKAGE_PATH, 
            config.DATA_PATH, 
            config.DATA)
    
    if args.merge_training_dev:
        merge_training_dev(
            config.DATA_PATH, 
            config.DATA)

