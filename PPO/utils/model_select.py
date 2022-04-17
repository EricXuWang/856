

from models.Nets import TwoNN, CNN, CifarNet
import torch

def model_selection(datasets, dataset_train, Model):
    img_size = dataset_train[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if datasets == 'CIFAR':
        num_fnn = 64 * 8 * 8
    else:
        num_fnn = 64 * 7 * 7


    if Model == 'TwoNN':
        net_glob = TwoNN(len_in).to(device)
    elif Model == 'CNN':
        net_glob = CNN(Channels = dataset_train[0][0].shape[0], num_fnn = num_fnn).to(device)
    elif Model == 'CifarNet':
        net_glob = CifarNet().to(device)

    # total = sum([param.nelement() for param in net_glob.parameters()])
    # print('Total Number of Parameters {}'.format(total))
    return net_glob
