import json


def get_config_from_json(path):
    
    with open(path, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict 
