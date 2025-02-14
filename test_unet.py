import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from data_loader import MetadataSegmentationDataset
from unet_model import UNetWithBiFPN
# from unet_model import UNet
import datetime
import multiprocessing

def main():
    # Define directories and file paths
    images_dir = 'dataset/images'
    masks_dir = 'dataset/annotations'
    csv_file = 'dataset/train.csv'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_output_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    model_path = 'best_unet_model.pth'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load the testing dataset
    test_dataset = MetadataSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        csv_file=csv_file,
        split='Training',
        transform=transform
    )

    # Define DataLoader
    num_workers = min(4, multiprocessing.cpu_count() // 2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,  # Changed batch size to 4
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = UNetWithBiFPN(in_channels=3, out_channels=1)
    # model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Define metrics (Updated to handle batch sizes)
    def dice_coefficient(pred, target):
        """Calculate Dice Coefficient for a batch."""
        batch_size = pred.size(0)
        dice_scores = []
        for i in range(batch_size):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            intersection = (pred_flat * target_flat).sum()
            dice = (2.0 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
            dice_scores.append(dice.item())
        return np.mean(dice_scores)

    def mean_iou(pred, target):
        """Calculate Mean IoU for a batch."""
        batch_size = pred.size(0)
        iou_scores = []
        for i in range(batch_size):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum() - intersection
            iou = intersection / (union + 1e-6)
            iou_scores.append(iou.item())
        return np.mean(iou_scores)

    def mean_pixel_accuracy(pred, target):
        """Calculate Mean Pixel Accuracy for a batch."""
        batch_size = pred.size(0)
        accuracies = []
        for i in range(batch_size):
            correct = (pred[i].int() == target[i].int()).sum()
            total = pred[i].numel()
            accuracy = correct.float() / total
            accuracies.append(accuracy.item())
        return np.mean(accuracies)
    
    def normalize_mask(mask):
        """Normalize mask to the range [0, 1]."""
        min_val = torch.min(mask)
        max_val = torch.max(mask)
        return (mask - min_val) / (max_val - min_val)
    # Initialize metrics
    total_dice, total_iou, total_pixel_accuracy, num_samples = 0.0, 0.0, 0.0, 0

    # Iterate over the testing dataset
    for idx, (image, mask, mask_ratio) in enumerate(test_loader):
        image, mask = image.to(device), mask.to(device)
        # mask = mask.unsqueeze(1) * 255 # Add channel dimension if necessary
        mask = mask.unsqueeze(1) # Add channel dimension if necessary
        mask = normalize_mask(mask)

        # Predict the mask
        with torch.no_grad():
            pred_mask = model(image)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.1).float()  # Adjust threshold as needed

        # Compute metrics for the batch (Updated to handle batch size)
        dice = dice_coefficient(pred_mask, mask)
        iou = mean_iou(pred_mask, mask)
        pixel_accuracy = mean_pixel_accuracy(pred_mask, mask)

        total_dice += dice * image.size(0)  # Weight by batch size
        total_iou += iou * image.size(0)
        total_pixel_accuracy += pixel_accuracy * image.size(0)
        num_samples += image.size(0)  # Update by batch size

        # Visualization for the first image in the batch
        single_image = image[0].squeeze().permute(1, 2, 0).cpu().numpy()
        single_mask = mask[0].squeeze().cpu().numpy()
        single_pred_mask = pred_mask[0].squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(single_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(single_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(single_pred_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        output_path = os.path.join(output_dir, f"test_image_{idx + 1}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Calculate average metrics
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_pixel_accuracy = total_pixel_accuracy / num_samples

    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Mean IoU: {avg_iou:.4f}")
    print(f"Average Mean Pixel Accuracy: {avg_pixel_accuracy:.4f}")

if __name__ == '__main__':
    main()

