from torchvision import datasets, transforms
from utils.preprocess import generate_iid_data, generate_non_iid_Half, generate_non_iid_data


def data_selection(dataset, data_distribution, non_iid_level,num_users):
    if dataset == 'mnist':
        # Calculate the scaled mean and std of the dataset in order to normalize the dataset.
        # It is noted that the python will not conduct normalization until the Dataloader works
        # In terms of the path, ../ represents previous level folder, ./ represents the current folder
        trainset = datasets.MNIST('../data/mnist/', train=True, download=True)
        mean = trainset.data.float().mean() / 255
        std = trainset.data.float().std() / 255
        # Pictures in the mnist are gray images with one channel
        # Normalize with mean 0.1307 and std 0.3081, (0.1307,) means that it is a tuple with one element
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((mean,), (std,))]))
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((mean), (std,))]))

    elif dataset == 'CIFAR':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

    elif dataset == 'Fashionmnist':
        trainset = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True)
        mean = trainset.data.float().mean() / 255
        std = trainset.data.float().std() / 255

        dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((mean,), (std,))]))
        dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Normalize((mean), (std,))]))

    if data_distribution == 'IID':
        dict_users = generate_iid_data(dataset_train, num_users)
    elif data_distribution == 'Non_IID':
        dict_users = generate_non_iid_data(dataset_train, num_users, non_iid_level)
    elif data_distribution == 'Non_IID_two_classes':
        dict_users = generate_non_iid_Half(dataset_train, num_users)

    return dataset_train, dataset_test, dict_users