import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Define a custom Dataset for Oxford Pet images
class OxfordPetDataset(Dataset):
    def __init__(self, csv_file, img_dir, split, transform=None):
        """
        :param csv_file: Path to the CSV file containing image names and split labels.
        :param img_dir: Directory where images are stored.
        :param split: Which split to use ("train", "val", or "test").
        :param transform: Transformations to apply to each image.
        """
        self.df = pd.read_csv(csv_file)
        # Filter rows based on the desired split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image identifier; assume the CSV has a column 'image_id'
        img_id = self.df.loc[idx, 'image_id']
        # Construct the full image path; adjust the extension as needed (e.g., .jpg)
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Assume there is also a label column; adjust if needed.
        label = self.df.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation for training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# For validation and testing, you typically only resize and normalize.
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_datasets():
    # Path to the CSV file that defines the splits
    csv_file = "oxford_pet_split.csv"
    # Path to the folder containing images (adjust as needed)
    img_dir = "path/to/oxford_pet_images"

    # Create dataset instances for each split
    train_dataset = OxfordPetDataset(csv_file, img_dir, split="train", transform=train_transform)
    val_dataset = OxfordPetDataset(csv_file, img_dir, split="val", transform=val_test_transform)
    test_dataset = OxfordPetDataset(csv_file, img_dir, split="test", transform=val_test_transform)
    
    return train_dataset, val_dataset, test_dataset

# Usage example:
train_set, val_set, test_set = get_datasets()
