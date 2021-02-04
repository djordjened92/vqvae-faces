import os
import glob
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(123)

class FaceImages(Dataset):

    def __init__(self, path_pattern, transform=None):
        self.transform = transform
        self.root_paths = glob.glob(path_pattern)
    
    def __len__(self):
        return len(self.root_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = io.imread(self.root_paths[idx]).astype('float32')
        image = image / 255.
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
def create_dataloader(images_path, image_w, image_h, batch_size, shuffle, workers):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((image_h, image_w))]
    )
    dataset = FaceImages(images_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

    return data_loader

if __name__=='__main__':

    # Test class implementation
    config = {
        "IMAGE_HEIGHT": 124,
        "IMAGE_WIDTH": 108,
        "BATCH_SIZE": 4,
        "SHUFFLE": False,
        "NUM_WORKERS": 0
    }
    path = '../data/kinship_ver_t1/test-faces/*.jpg'

    data_loader = create_dataloader(path, config['IMAGE_WIDTH'], config['IMAGE_HEIGHT'], config['BATCH_SIZE'], config['SHUFFLE'], config['NUM_WORKERS'])

    print(iter(data_loader).next())
