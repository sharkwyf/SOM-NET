import torch
from config import get_config
from trainer import Trainer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from mydataset import MyDataset, paired_collate_fn


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def showpic(image):
    image = image.reshape(3, 32, 32)
    image = np.stack((image[0], image[1], image[2]), axis=2)
    plt.imshow(image)
    plt.show()
    pass


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


if __name__ == "__main__":
    config = get_config()
    
    # Load data
    download = True
    need_shuffle = True
    images, labels = [], []
    for i in range(5):
        content = unpickle(f'./data/data_batch_{i + 1}')
        images.append(content[b"data"])
        labels.append(content[b"labels"])
    train_dataset = MyDataset(images, labels)
    images, labels = [], []
    content = unpickle(f'./data/test_batch')
    images.append(content[b"data"])
    labels.append(content[b"labels"])
    test_dataset = MyDataset(images, labels)
    
    train_loader = DataLoader(train_dataset, shuffle=need_shuffle,
                              batch_size=config.batch_size, collate_fn=paired_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=need_shuffle,
                             batch_size=config.batch_size, collate_fn=paired_collate_fn)
    
    trainer = Trainer(config)
    trainer.train(train_loader, test_loader)
    trainer.test(test_loader)
    
    pass