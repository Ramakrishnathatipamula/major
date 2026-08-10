from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Standard ImageNet RGB Mean and Standard Deviation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(image_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """
    Define image preprocessing and normalization transforms.
    
    Args:
        image_size: Tuple of (height, width) for resizing input images.
        
    Returns:
        Dictionary containing 'train' and 'val' transform pipelines.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return {
        "train": transform_pipeline,
        "val": transform_pipeline,
        "test": transform_pipeline
    }

def load_plant_dataset(data_dir: Path, transform=None) -> datasets.ImageFolder:
    """
    Load plant leaf images using PyTorch ImageFolder.
    
    Args:
        data_dir: Path to directory containing class subfolders.
        transform: Torchvision transform pipeline to apply to each image.
        
    Returns:
        torchvision.datasets.ImageFolder instance.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory '{data_path}' does not exist.")
        
    if transform is None:
        transform = get_transforms()["val"]
        
    dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    return dataset

def get_class_names(dataset: datasets.ImageFolder) -> List[str]:
    """Retrieve list of class names detected by ImageFolder."""
    return dataset.classes

def get_class_to_idx(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """Retrieve mapping from class string names to integer class indices."""
    return dataset.class_to_idx

def create_data_loader(
    dataset: datasets.ImageFolder, 
    batch_size: int = 4, 
    shuffle: bool = True
) -> DataLoader:
    """
    Wrap an ImageFolder dataset in a PyTorch DataLoader for batch processing.
    
    Args:
        dataset: PyTorch ImageFolder dataset.
        batch_size: Number of images per mini-batch.
        shuffle: Whether to shuffle images every epoch.
        
    Returns:
        torch.utils.data.DataLoader instance.
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0  # Safe default for cross-platform compatibility
    )
