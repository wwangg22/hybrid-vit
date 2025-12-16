"""
Train hybrid Vision Transformer models.

Trains ViT models on MNIST and CIFAR-10. Supports different attention patterns
and tracks how long training/inference takes.

Usage:
    python train.py --config configs/vanilla_vit_mnist.yml
    python train.py --config configs/alternating_vit_cifar10.yml
"""

import os
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import wandb
from PIL import Image

from transformers import (
    VanillaViT,
    AlternatingViT,
    PerformerFirstViT,
    RegularFirstViT,
    CustomPatternViT
)


class ImageDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace datasets so they work with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        # Check if this is actually a HuggingFace dataset
        self.is_hf = hasattr(hf_dataset, '__getitem__') and not isinstance(hf_dataset, torch.utils.data.Dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.is_hf:
            item = self.dataset[idx]
            image = item['image']
            label = item['label']
            
            # Make sure image is RGB
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        else:
            return self.dataset[idx]


def load_config(config_path):
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_transforms(img_size, dataset_name):
    """Get the right transforms for training and testing."""
    
    if dataset_name.lower() == 'mnist':
        # MNIST is grayscale, so convert to RGB by repeating the channel 3 times
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 is already RGB, so just do standard augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform


def load_data(config):
    """Load the dataset and create data loaders."""
    
    dataset_name = config['data']['dataset']
    batch_size = config['training']['batch_size']
    img_size = config['model']['img_size']
    
    if dataset_name.lower() == 'mnist':
        try:
            dataset = load_dataset("ylecun/mnist", trust_remote_code=True)
        except:
            # If HuggingFace fails, just use torchvision
            print("Loading MNIST from torchvision instead...")
            from torchvision.datasets import MNIST
            train_transform, test_transform = get_transforms(img_size, dataset_name)
            
            train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
            test_dataset = MNIST(root='./data', train=False, download=True, transform=test_transform)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=True
            )
            
            return train_loader, test_loader
            
    elif dataset_name.lower() == 'cifar10':
        try:
            dataset = load_dataset("cifar10", trust_remote_code=True)
        except:
            # Same fallback for CIFAR-10
            print("Loading CIFAR-10 from torchvision instead...")
            from torchvision.datasets import CIFAR10
            train_transform, test_transform = get_transforms(img_size, dataset_name)
            
            train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=True
            )
            
            return train_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # If we got here, HuggingFace worked
    train_transform, test_transform = get_transforms(img_size, dataset_name)
    
    train_dataset = ImageDataset(dataset['train'], transform=train_transform)
    test_dataset = ImageDataset(dataset['test'], transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, test_loader


def create_model(config, device):
    """Build the model from config."""
    
    model_type = config['model']['type']
    model_params = config['model']
    
    # Parameters that all models need
    common_params = {
        'img_size': model_params['img_size'],
        'patch_size': model_params['patch_size'],
        'in_channels': model_params['in_channels'],
        'num_classes': model_params['num_classes'],
        'dim': model_params['dim'],
        'num_heads': model_params['num_heads'],
        'ff_hidden_dim': model_params['ff_hidden_dim'],
        'dropout': model_params['dropout'],
    }
    
    # Pick the right model
    if model_type == 'vanilla':
        model = VanillaViT(
            **common_params,
            num_layers=model_params['num_layers']
        )
    
    elif model_type == 'alternating':
        model = AlternatingViT(
            **common_params,
            num_layers=model_params['num_layers'],
            nb_features=model_params.get('nb_features', None)
        )
    
    elif model_type == 'performer_first':
        model = PerformerFirstViT(
            **common_params,
            num_layers=model_params['num_layers'],
            num_performer_layers=model_params['num_performer_layers'],
            nb_features=model_params.get('nb_features', None)
        )
    
    elif model_type == 'regular_first':
        model = RegularFirstViT(
            **common_params,
            num_layers=model_params['num_layers'],
            num_regular_layers=model_params['num_regular_layers'],
            nb_features=model_params.get('nb_features', None)
        )
    
    elif model_type == 'custom_pattern':
        layer_pattern = model_params['layer_pattern']
        model = CustomPatternViT(
            **common_params,
            layer_pattern=layer_pattern,
            nb_features=model_params.get('nb_features', None)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Show how big the model is
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_type}")
    print(f"Parameters: {num_params:,}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Run one training epoch."""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
 
        # Backward pass
        loss.backward()
        optimizer.step()
        # Track stats
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Test the model and time how long inference takes."""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track how long inference takes
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            # Time this batch
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    
    return avg_loss, accuracy, avg_inference_time


def train(config, device):
    """Main training loop."""
    
    # Make output folder
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start wandb tracking
    wandb.init(
        project=config.get('wandb', {}).get('project', 'hybrid-vit'),
        name=config.get('wandb', {}).get('run_name', f"{config['model']['type']}_{config['data']['dataset']}"),
        config=config,
        dir=output_dir,
        tags=[config['model']['type'], config['data']['dataset']],
        notes=config.get('wandb', {}).get('notes', '')
    )
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = load_data(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Build model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Let wandb watch the model
    wandb.watch(model, log='all', log_freq=100)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate goes down over time
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Train!
    print("\nStarting training...")
    best_acc = 0
    training_times = []
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train one epoch
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        
        # Test it
        test_loss, test_acc, inference_time = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log everything to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'test/loss': test_loss,
            'test/accuracy': test_acc,
            'time/epoch_time': epoch_time,
            'time/inference_time': inference_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print results
        print(f'\nEpoch {epoch}/{config["training"]["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s, Inference Time: {inference_time:.4f}s')
        
        # Save if this is the best so far
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
        
        # Save checkpoints periodically
        if epoch % config['training'].get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Done! Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Average Training Time per Epoch: {np.mean(training_times):.2f}s")
    print(f"Total Training Time: {np.sum(training_times):.2f}s")
    
    # Save summary to wandb
    wandb.run.summary['best_test_acc'] = best_acc
    wandb.run.summary['final_test_acc'] = test_acc
    wandb.run.summary['avg_training_time_per_epoch'] = np.mean(training_times)
    wandb.run.summary['total_training_time'] = np.sum(training_times)
    wandb.run.summary['num_parameters'] = sum(p.numel() for p in model.parameters())
    
    # Save final model
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'config': config,
        'best_acc': best_acc,
        'avg_training_time': np.mean(training_times),
        'total_training_time': np.sum(training_times)
    }, output_dir / 'final_model.pth')
    
    # Upload model to wandb
    artifact = wandb.Artifact(
        name=f"{config['model']['type']}_{config['data']['dataset']}_model",
        type='model',
        description=f"Final model checkpoint for {config['model']['type']} on {config['data']['dataset']}"
    )
    artifact.add_file(str(output_dir / 'final_model.pth'))
    wandb.log_artifact(artifact)
    
    # Save results to file
    results = {
        'model_type': config['model']['type'],
        'dataset': config['data']['dataset'],
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'avg_training_time_per_epoch': np.mean(training_times),
        'total_training_time': np.sum(training_times),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Figure out what device to use
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed so results are reproducible
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['seed'])
    
    # Go!
    train(config, device)


if __name__ == '__main__':
    main()