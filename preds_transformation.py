import json
import config

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
        with open(logs_path + '/' + task + '_#####_test_#####' + '_ensemles' + '.json', "w") as f:
            json.dump(models_preds, f, indent=2)


def round_to_closes_value(transformers, tasks, logs_path, ensemble=''):
    possible_values = [0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
    
    for task in tasks:
        # for model in transformers + [ensemble]:
        for model in [ensemble]:
            with open(logs_path + '/' + task + '_#####_test_#####_' + model + '.json', 'r') as f:
                json_preds = json.load(f)
                
            for index, preds in json_preds.items():
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
    ###### I STOP HERE !!!!!!!
        # - Submet :
        #     1) best LLM
        #     2) Ensemble
        #     2) Ensemble + round to closes value
    
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