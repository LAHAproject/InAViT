import os
import torch
import json

def writetestjson(test_score_dict, cfg):
    json_file = os.path.join(cfg.OUTPUT_DIR, 'test.json')
    verb_num_classes = 97
    noun_num_classes = 300
    predictions = {}
    if cfg.EPICKITCHENS.ANTICIPATION:
        challenge = 'action_anticipation'
    else:
        challenge = 'action_recognition'
    predictions = {'version': '0.2',\
                'challenge': challenge,
                'results': {}}
    predictions['sls_pt'] = 1
    predictions['sls_tl'] = 3
    predictions['sls_td'] = 3
    for nid in test_score_dict.keys():
        verb_scores = [ score_dict['verb'] for score_dict in test_score_dict[nid] ]
        noun_scores = [ score_dict['noun'] for score_dict in test_score_dict[nid] ]
        verb_score = torch.mean(torch.stack(verb_scores), dim=0)
        noun_score = torch.mean(torch.stack(noun_scores), dim=0)

        predictions['results'][str(nid)] = {}
        predictions['results'][str(nid)]['verb'] = {str(ii): round(float(verb_score[ii]),5) for ii in range(verb_num_classes)}
        predictions['results'][str(nid)]['noun'] = {str(ii): round(float(noun_score[ii]),5) for ii in range(noun_num_classes)}
            
    with open(json_file, 'w') as fp:
        json.dump(predictions, fp,  indent=4) 
    
    return None