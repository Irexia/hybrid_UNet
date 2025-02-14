import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import MetadataSegmentationDataset

from unet_model import UNetWithBiFPN
import torch.nn.functional as F



def dice_coefficient(pred, target):
        """Calculate Dice Coefficient."""
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)


# Weighted BCE Loss with Dynamic Small Region Threshold
def weighted_bce_loss(preds, targets, mask_ratios, weight_multiplier= 1.5):
    preds = torch.sigmoid(preds)  # Apply sigmoid to convert logits to probabilities
    weights = torch.ones_like(targets, device=targets.device)  # Initialize weights to 1
    binary_mask = targets > 0.5  # Foreground mask

    # Calculate total pixels per mask
    batch, _, height, width = targets.size()
    total_pixels = height * width

    # Dynamically calculate small region threshold based on mask_ratios
    small_region_threshold = (mask_ratios.mean() * total_pixels).item()

    # Adjust weights for small regions
    flat_indices = torch.arange(batch * height * width, device=targets.device).reshape(batch, height, width)
    labeled_mask = binary_mask * flat_indices + (~binary_mask) * 0
    unique_labels, counts = torch.unique(labeled_mask, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if count < small_region_threshold:
            weights[labeled_mask == label] *= weight_multiplier

    # Use mask_ratio to further adjust weights
    weights *= (1.5 + (1.0 - mask_ratios).unsqueeze(1).unsqueeze(2))

    return F.binary_cross_entropy(preds, targets, weight=weights, reduction='mean')


# Directories and Paths

images_dir = 'dataset/images'
masks_dir = 'dataset/annotations'
csv_file = 'dataset/train.csv'
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Parameters
batch_size = 16
num_epochs = 22


# Create a directory based on parameters
experiment_dir = f"###final {batch_size}_{num_epochs}_hybrid_UNet_mulitplier1.5_maskratio1.5"
os.makedirs(experiment_dir, exist_ok=True)

model_save_path = f'best_unet_model.pth'

# Transformations with Augmentations
image_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(30),  # Random rotation between -30 to 30 degrees
    # transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1)),  # Random affine transformation
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(30),  # Random rotation
    # transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1)),  # Random affine
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Datasets and Loaders
train_dataset = MetadataSegmentationDataset(images_dir, masks_dir, csv_file, split='Training', transform= image_transform)
val_dataset = MetadataSegmentationDataset(images_dir, masks_dir, csv_file, split='Validation', transform= mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
# model = UNet(in_channels=3, out_channels=1).to(device)
model = UNetWithBiFPN(in_channels=3, out_channels=1).to(device)
criterion = weighted_bce_loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
train_losses, val_losses, val_dice_scores = [], [], []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss, train_dice = 0.0, 0.0

    for images, masks, mask_ratios in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        images, masks, mask_ratios = images.to(device), masks.to(device), mask_ratios.to(device)
        # masks = normalize_mask(masks)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks, mask_ratios)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_dice += dice_coefficient(torch.sigmoid(outputs), masks).item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_dice = train_dice / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks, mask_ratios in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
            images, masks, mask_ratios = images.to(device), masks.to(device), mask_ratios.to(device)
            # masks = normalize_mask(masks)
            outputs = model(images)
            loss = criterion(outputs, masks, mask_ratios)

            val_loss += loss.item() * images.size(0)
            val_dice += dice_coefficient(torch.sigmoid(outputs), masks).item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_dice = val_dice / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{model_save_path}_epoch{epoch + 1}_val_loss{avg_val_loss:.4f}.pth")
        print(f"Best model{epoch} saved to {model_save_path}")

    # Save Predicted Mask Example
    sample_image = images[0].detach().cpu()
    sample_mask = masks[0].detach().cpu()
    with torch.no_grad():
        sample_image = sample_image.unsqueeze(0).to(device)
        predicted_mask = model(sample_image)
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = (predicted_mask > 0.1).float()

    # Plot and save
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sample_mask.squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze().cpu(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    output_path = os.path.join(output_dir, f"epoch_{epoch + 1}_prediction.png")
    plt.savefig(output_path)
    plt.close()

# Save Training/Validation Loss and Dice Plots
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(experiment_dir, "loss_curves.png")
plt.savefig(loss_curve_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), val_dice_scores, label="Validation Dice Score")
plt.title("Validation Dice Scores")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.grid(True)
dice_score_path = os.path.join(experiment_dir, "dice_scores.png")
plt.savefig(dice_score_path)
plt.show()

print("Training Complete.")

