import json
import config
from itertools import combinations_with_replacement, permutations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_cosine_similarity(list1, set_of_lists):
    list1 = np.array(list1).reshape(1, -1)  # Reshape list1 to a 2D array
    set_of_lists = np.array(list(set_of_lists))  # Convert the set of lists to an array

    similarities = cosine_similarity(list1, set_of_lists)  # Calculate cosine similarity

    most_similar_index = np.argmax(similarities)  # Get the index of the most similar list
    most_similar_list = set_of_lists[most_similar_index]  # Get the most similar list

    return most_similar_list


def remove_identical_lists(list_of_lists):
    unique_lists = []

    for sublist in list_of_lists:
        # Convert the sublist to a tuple for hashability
        sublist_tuple = tuple(sublist)

        if sublist_tuple not in unique_lists:
            unique_lists.append(sublist_tuple)

    # Convert the unique lists back to lists
    unique_lists = [list(sublist) for sublist in unique_lists]

    return unique_lists

def generate_possible_comb(task):
    value_list = [0.0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
    combination_length = {'task1': 2, 'task2': 4, 'task3': 6}
    valid_combinations = []
    perm_list = []
    
    combinations = combinations_with_replacement(value_list, combination_length[task])
    for combination in combinations:
        if task != 'task3':
            if sum(combination) == 1:
                valid_combinations.append(list(combination))
        else:
            valid_combinations.append(list(combination))
            
    for list_values in valid_combinations:
        permuts = permutations(list_values, combination_length[task])
        for permut in permuts:
            perm_list.append(list(permut))
            
    return remove_identical_lists(perm_list)

def sample_close_values(dict_preds , possible_preds):
    pred_tuples = list(dict_preds.items())
    value_list = []
    key_list = []
    for item in pred_tuples:
        value_list.append(item[1])
        key_list.append(item[0])
    
    most_similar_pred = calculate_cosine_similarity(value_list, possible_preds)
    
    new_preds = {}
    for i in range(len(dict_preds)):
        new_preds[key_list[i]] = most_similar_pred[i]
        
    return new_preds

def ensemble(transformers, tasks, logs_path):
    for task in tasks:
        list_json_preds = []
        # Load the JSON file
        for transformer in transformers:
            with open(logs_path + '/' + task + '_#####_test_#####_' + transformer + '.json', "r") as f:
                list_json_preds.append(json.load(f))
        
        # sum preds
        for i, model_n_preds in enumerate(list_json_preds):
            if i == 0:
                models_preds = model_n_preds
            else:
                for index, val in model_n_preds.items():
                    models_preds[index]['soft_label'] = {j: v + val['soft_label'][j] for j, v in models_preds[index]['soft_label'].items()}
        
        # average preds
        for index, val in models_preds.items():
            models_preds[index]['soft_label'] = {j: v / len(transformers) for j, v in models_preds[index]['soft_label'].items()}
            
        # Save the dictionary as a JSON file
        with open(logs_path + '/' + task + '_#####_test_#####' + '_ensemble' + '.json', "w") as f:
            json.dump(models_preds, f, indent=2)
                

def round_to_closes_value(transformers, tasks, logs_path, ensemble=''):
    for task in tasks:
        if task != 'task3':
            possible_values = generate_possible_comb(task)
        else:
            possible_values = [0.0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
        
        # for model in transformers + [ensemble]:
        for model in [ensemble]:
            with open(logs_path + '/' + task + '_#####_test_#####_' + model + '.json', 'r') as f:
                json_preds = json.load(f)
                
            for index, preds in json_preds.items():
                    if task != 'task3':
                        json_preds[index]['soft_label'] = sample_close_values(preds['soft_label'], possible_values)
                    else:
                        json_preds[index]['soft_label'] = {j: min(possible_values, key=lambda x: abs(x - v)) for j, v in preds['soft_label'].items()}
                    
            # Save the dictionary as a JSON file
            with open(logs_path + '/' + task + '_#####_test_#####_' + model + '_round_to_closes_value' +'.json', "w") as f:
                json.dump(json_preds, f, indent=2)


def get_hard_preds(transformers, tasks, logs_path, ensemble=''):
    for task in tasks:
        for model in transformers + [ensemble] + [ensemble + '_round_to_closes_value']:
            with open(logs_path + '/' + task + '_#####_test_#####_' + model + '.json', 'r') as f:
                json_preds = json.load(f)

            for index, preds in json_preds.items():
                if task in ['task1', 'task2']:
                    json_preds[index]['hard_label'] = max(preds['soft_label'], key=preds['soft_label'].get)
                
                else:
                    if preds['soft_label']["NO"] > 0.84:
                        json_preds[index]['hard_label'] = ["NO"]
                    
                    else:
                        json_preds[index]['hard_label'] = []
                        for label, value in  preds['soft_label'].items():
                            if value > ((1 - preds['soft_label']["NO"]) / 2) and label != "NO":
                                json_preds[index]['hard_label'].append(label) 

            # Save the dictionary as a JSON file
            with open(logs_path + '/' + task + '_#####_test_#####_' + model + '_plus_hard-preds' +'.json', "w") as f:
                json.dump(json_preds, f, indent=2)
        
if __name__ == "__main__":    
    ensemble(config.TRANSFORMERS, 
                    config.LABELS, 
                    config.LOGS_PATH)
    
    
    round_to_closes_value(config.TRANSFORMERS, 
                            config.LABELS, 
                            config.LOGS_PATH,
                            'ensemble')
    
    
    get_hard_preds(config.TRANSFORMERS, 
                        config.LABELS, 
                        config.LOGS_PATH,
                        'ensemble')
    
