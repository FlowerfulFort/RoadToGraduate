import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob

class RoadDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

def create_dataloader(train_dir, label_dir, batch_size, transform):
    image_paths = sorted(glob.glob(os.path.join(train_dir, '*.png')))
    label_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    dataset = RoadDataset(image_paths, label_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
