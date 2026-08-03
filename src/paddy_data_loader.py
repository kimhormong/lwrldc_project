from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_train_val_test_data(batch_size=64, num_workers=4):
    
    train_dir = "../data/processed/paddy_resized_80_10_10/train"
    val_dir = "../data/processed/paddy_resized_80_10_10/val"
    test_dir = "../data/processed/paddy_resized_80_10_10/test"


    transforms_all = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transforms_all)
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms_all)
    test_dataset = datasets.ImageFolder(test_dir, transform=transforms_all)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,

    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )


    return train_loader, val_loader, test_loader



