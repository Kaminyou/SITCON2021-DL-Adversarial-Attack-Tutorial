import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ImageNetteDataset(Dataset):
    def __init__(self, mapping_folder_to_label, data_root = "./data/imagenette2/train", train = True, return_file_name=False, return_folder_name=False, simple_transform=False):
        self.data_root = data_root
        self.mapping_folder_to_label = mapping_folder_to_label
        self.train = train
        self.return_file_name = return_file_name
        self.return_folder_name = return_folder_name
        if simple_transform:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.Resize(256), 
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
        self.init_dataset()
    
    def init_dataset(self):
        self.data = []
        self.labels = []
        for folder in os.listdir(self.data_root):
            for image_file in os.listdir(os.path.join(self.data_root, folder)):
                image_path = os.path.join(self.data_root, folder, image_file)
                label = self.mapping_folder_to_label[folder]
                
                self.data.append(image_path)
                self.labels.append(label)
                
    def __read_img(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return img
    
    def __getitem__(self, index):
        image_path = self.data[index]
        label = self.labels[index]
        img = self.__read_img(image_path)
        img = self.transform(img)
        
        if self.return_file_name:
            file_name = os.path.basename(image_path)
            return img, label, file_name
        elif self.return_folder_name:
            folder_name = os.path.basename(os.path.basename(image_path))
            return img, label, folder_name
        else:
            return img, label
        
    def __len__(self):
        return len(self.labels)