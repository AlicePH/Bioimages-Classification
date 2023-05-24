import torch

def get_optimizer(opt_name):
    if opt_name=='sgd':
        optimizer = torch.optim.SGD
        
    elif opt_name=='adam':
        optimizer = torch.optim.Adam
        
    elif opt_name=='adamw':
        optimizer = torch.optim.AdamW
    else:
        raise ValueError('Incorrect optimizer name')
    
    return optimizer