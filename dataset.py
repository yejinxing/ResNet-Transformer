import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class CardiacDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.png'):
                    self.samples.append((os.path.join(class_dir, file), self.class_to_idx[class_name]))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                 std=[0.229])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), label

def create_loaders(data_dir, batch_size=32, seed=42):
    full_dataset = CardiacDataset(data_dir)
    total = len(full_dataset)
    
    # 计算各数据集尺寸
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size
    
    # 设置随机种子保证可重复性
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader