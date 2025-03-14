import os
import torch
import math
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, ConcatDataset, DataLoader, Dataset
import numpy as np

def build_transform(input_size=224, interpolation="bicubic",
                    mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261),
                    crop_pct=0.9):
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR
    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size, interpolation=ip
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset(args, config):
    if args.dataset == 'CIFAR10':
        transform = build_transform(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261), crop_pct=0.9)
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        full_dataset = ConcatDataset([train_set, test_set])
        
    elif args.dataset == 'CIFAR100':
        transform = build_transform(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761), crop_pct=0.9)
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        full_dataset = ConcatDataset([train_set, test_set])
        
    elif args.dataset == 'CINIC10':
        transform = build_transform(mean=(0.47889522, 0.47227842, 0.43047404), std=(0.24205776, 0.23828046, 0.25874835), crop_pct=0.9)
        full_dataset = ConcatDataset([datasets.ImageFolder(config.path.data_path+'train', transform), 
                                      datasets.ImageFolder(config.path.data_path+'test', transform),
                                      datasets.ImageFolder(config.path.data_path+'val', transform)]) 
    
    elif args.dataset == 'GTSRB':
        transform = build_transform(mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669), crop_pct=0.9)
        full_dataset = ConcatDataset([datasets.ImageFolder(config.path.data_path+'train', transform), 
                                      datasets.ImageFolder(config.path.data_path+'test', transform),]) 
        
    elif args.dataset == 'ImageNet100':
        transform = transforms.Compose([transforms.Resize([32, 32], Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        full_dataset = ConcatDataset([datasets.ImageFolder(config.path.data_path+'train', transform), 
                                      datasets.ImageFolder(config.path.data_path+'test', transform),
                                      datasets.ImageFolder(config.path.data_path+'val', transform)]) 

    
    def get_class_indices(concat_dataset):
        class_indices = {i: [] for i in range(config.general.num_classes)}
        offset = 0
        for ds in concat_dataset.datasets:
            if hasattr(ds, 'targets'):
                ds_targets = np.array(ds.targets)
            else:
                ds_targets = np.array([s[1] for s in ds.samples])
            for class_id in range(config.general.num_classes):
                indices = np.nonzero(ds_targets == class_id)[0]
                if indices.size > 0:
                    class_indices[class_id].extend((indices + offset).tolist())
            offset += len(ds)
        return class_indices

    class_indices = get_class_indices(full_dataset)

    target_train_indices = []
    target_test_indices = []
    shadow_train_indices = []
    shadow_test_indices = []
    ditill_indices = []
    
    for class_id in range(config.general.num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        if args.dataset == 'GTSRB':
            num = len(indices) // 5
            target_train_indices.extend( indices[: num] )
            target_test_indices.extend( indices[num : num*2] )
            shadow_train_indices.extend( indices[num*2 : num*3] )
            shadow_test_indices.extend( indices[num*3 : num*4] )
            ditill_indices.extend( indices[num*4 : ] )
        else:
            target_train_indices.extend( indices[: config.general.num_per_class] )
            target_test_indices.extend( indices[config.general.num_per_class : config.general.num_per_class*2] )
            shadow_train_indices.extend( indices[config.general.num_per_class*2 : config.general.num_per_class*3] )
            shadow_test_indices.extend( indices[config.general.num_per_class*3 : config.general.num_per_class*4] )
            if args.dataset == 'CINIC10':
                ditill_indices.extend( indices[config.general.num_per_class*4 : config.general.num_per_class*6] )
            else:
                ditill_indices.extend( indices[config.general.num_per_class*4 : ] )
                
    target_train_set = Subset(full_dataset, target_train_indices)
    target_test_set = Subset(full_dataset, target_test_indices)
    shadow_train_set = Subset(full_dataset, shadow_train_indices)
    shadow_test_set = Subset(full_dataset, shadow_test_indices)
    distill_dataset = Subset(full_dataset, ditill_indices)
    
    train_loader_target = DataLoader(
        target_train_set, 
        batch_size=config.learning.train_batch_size, 
        shuffle=True, 
        num_workers=config.learning.num_workers,
        pin_memory=True)
    test_loader_target = DataLoader(
        target_test_set, 
        batch_size=config.learning.test_batch_size, 
        shuffle=False, 
        num_workers=config.learning.num_workers,
        pin_memory=True)
    dataloaders_target = {"train": train_loader_target, "test": test_loader_target}
    dataset_sizes_target = {"train": len(target_train_set), "test": len(target_test_set)}
    
    train_loader_shadow = DataLoader(
        shadow_train_set, 
        batch_size=config.learning.train_batch_size, 
        shuffle=True, 
        num_workers=config.learning.num_workers,
        pin_memory=True)
    test_loader_shadow = DataLoader(
        shadow_test_set, 
        batch_size=config.learning.test_batch_size, 
        shuffle=False, 
        num_workers=config.learning.num_workers,
        pin_memory=True)
    dataloaders_shadow = {"train": train_loader_shadow, "test": test_loader_shadow}
    dataset_sizes_shadow = {"train": len(shadow_train_set), "test": len(shadow_test_set)}
    
    dataloader_distill = DataLoader(
        distill_dataset, 
        batch_size=config.distill.train_batch_size, 
        shuffle=True, 
        num_workers=config.distill.num_workers,
        pin_memory=True)
    
    return dataloaders_target, dataset_sizes_target, dataloaders_shadow, dataset_sizes_shadow, dataloader_distill
