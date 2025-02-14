import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class MetadataSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_file, split='Training', transform=None):
        """
        A custom dataset for segmentation tasks with metadata from a CSV file.

        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            csv_file (str): Path to the CSV file with metadata.
            split (str): Dataset split to use ('Training' or 'Validation').
            transform (callable, optional): Transformations for images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Load and filter the CSV file
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame['Split'] == split]

        # Extract image and mask filenames
        self.image_names = self.data_frame['Name'].values
        self.mask_names = self.data_frame['Label file'].values

        # Include additional metadata if needed (e.g., mask ratio)
        if 'Mask_Ratio' in self.data_frame.columns:
            self.mask_ratios = self.data_frame['Mask_Ratio'].values
        else:
            self.mask_ratios = None

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, index):
        """
        Get a sample (image, mask, and metadata) by index.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple: (image, mask, metadata_tensor)
        """
        # Retrieve image and mask filenames
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        # Ensure correct file extensions
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_name += '.jpg'  # Add .jpg extension if missing

        if not mask_name.lower().endswith('.png'):
            mask_name += '.png'  # Add .png extension if missing

        # Get full paths for image and mask
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Load the image and mask
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Retrieve additional metadata (e.g., mask ratio) as a tensor
        if self.mask_ratios is not None:
            mask_ratio = torch.tensor([self.mask_ratios[index]], dtype=torch.float32)
        else:
            mask_ratio = torch.tensor([0.0], dtype=torch.float32)  # Default to 0 if not provided

        # Return image, mask, and metadata tensor
        return image, mask, mask_ratio


