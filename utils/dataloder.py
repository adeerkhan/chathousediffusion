import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 定义位置和大小的映射
location_map = {
    "north": 0,
    "northwest": 1,
    "west": 2,
    "southwest": 3,
    "south": 4,
    "southeast": 5,
    "east": 6,
    "northeast": 7,
    "center": 8
}

size_map = {
    "XL": 4,
    "L": 3,
    "M": 2,
    "S": 1,
    "XS": 0
}

# 定义房间类型映射
room_type_map = {
    "LivingRoom": 0,
    "MasterRoom": 1,
    "Kitchen": 2,
    "Bathroom": 3,
    "DiningRoom": 4,
    "ChildRoom": 5,
    "StudyRoom": 6,
    "SecondRoom": 7,
    "GuestRoom": 8,
    "Balcony": 9,
    "Entrance": 10,
    "Storage": 11
}


class FloorplanDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        """
        Args:
            json_dir (string): Directory containing JSON files.
            image_dir (string): Directory containing image files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        json_path = os.path.join(self.json_dir, json_file)
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        rooms = data['room']
        features = []

        for room_type, room_info in rooms.items():
            for room in room_info['rooms']:
                features.append([
                    room_type_map[room_type],
                    room_info['num'],
                    location_map[room['location']],
                    size_map[room['size']]
                ])
        
        if features:
            features = torch.tensor(features, dtype=torch.float32).flatten()
        else:
            features = torch.tensor([], dtype=torch.float32)  # Handle cases with no rooms

        image_file = json_file.replace('.json', '.png')
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, features
    

def get_dataloader(json_dir, image_dir, batch_size=4, shuffle=True, transform=None):
    dataset = FloorplanDataset(json_dir, image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataloader = get_dataloader('path/to/json', 'path/to/images', transform=transform)
    for images, features in dataloader:
        print(images.shape, features.shape)
