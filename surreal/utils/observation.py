import torch

def get_matching_keys_for_modality(obs, modality, input_config):
    '''
    Extracts observation keys that match the given modality
    Args:
        obs: an observation of type OrderedDict, or
        	an observation spec of type OrderedDict
        modality: an observation modality, such as 'pixel' or 'low_dim'
        model_config: the model config.  This function uses learner_config.model.input.

        For example, if the config is
        	'input': {
                'pixel':['image'],
                'low_dim':['joint_pos', 'joint_vel'],
            }
        and modality is 'pixel'
        and the observation has keys ['image', 'joint_vel', 'joint_pos', 'x', 'y'],
        this function will return ['image']
    '''
    
    valid_keys = input_config[modality]
    # Iterate over keys manually to compute intersection of 
    # obs.keys() and valid_keys in order to preserve order
    matching_keys = []
    for key in obs.keys():
    	if key in valid_keys:
    		matching_keys.append(key)
    return matching_keys

def gather_low_dim_input(obs, input_config):
    matching_keys = get_matching_keys_for_modality(obs, 
                                                  'low_dim', 
                                                  input_config)
    list_obs_ld = [obs[key] for key in matching_keys] # technically here we should use the intersect
    if len(list_obs_ld) < 1: return None
    obs_low_dim = torch.cat(list_obs_ld, -1)
    return obs_low_dim