import torch


def save_pytorch_data(path, net, optimizer=None, additional_data=None):
    checkpoint = {'net': net.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if additional_data is not None:
        checkpoint['additional_data'] = additional_data
    
    torch.save(checkpoint, path)
    
def load_pytorch_data(path, net, optimizer=None):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return checkpoint
