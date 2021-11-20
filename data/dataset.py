"""Utils classes and functions."""
import cv2
import glob
import os

from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset


class MNIST_M(Dataset):
    """MNIST_M Dataset."""

    def __init__(self, pre_load_files, data_root, data_list, transform=None):
        """Initialize MNIST_M data set.

        Keyword Params:
            root -- the root folder containing the data
            data_list -- the file containing list of images - labels
            transform (optional) -- tranfrom images when loading
        """
        self.root = data_root
        self.transform = transform
        self.pre_load_files = pre_load_files
        
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()
        
        if self.pre_load_files == 1:
            self.images = []
            self.labels = []
            for idx in range(len(self.data_list)):
                if (idx % 100 == 0):
                    with open('dataset_temp.txt', 'a') as f:
                        f.write('Image '+str(idx+1)+'/'+str(len(self.data_list))+'\n')
            
                img_name, labels = self.data_list[idx].split()
                imgs = Image.open(os.path.join(self.root, img_name)).convert('RGB')

                if self.transform:
                    imgs = self.transform(imgs)
                    
                self.images.append(imgs)
                self.labels.append(int(labels))
                
            os.remove('dataset_temp.txt')

    def __len__(self):
        """Get length of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        if self.pre_load_files == 0:
            img_name, labels = self.data_list[idx].split()
            imgs = Image.open(os.path.join(self.root, img_name)).convert('RGB')

            if self.transform:
                imgs = self.transform(imgs)

            labels = int(labels)
        else:
            imgs = self.images[idx]
            labels = self.labels[idx]

        return imgs, labels


class SynSigns(Dataset):
    """Synthetic signalisation Dataset."""

    def __init__(self, pre_load_files, data_root, annotation_file, transform=None):
        """Init dataset.

        Keyword Parameters:
            data_root -- root of the dataset folder
            annotation_file -- name of the annotation file
            trainsform (optional) -- transform images when loading
        """
        path = os.path.join(data_root, annotation_file)
        self.data_root = data_root
        self.transform = transform
        self.pre_load_files = pre_load_files

        with open(path, 'r') as f:
            self.data_list = f.readlines()
        
        if self.pre_load_files == 1:
            self.images = []
            self.labels = []
            for idx in range(len(self.data_list)):
                if (idx % 100 == 0):
                    with open('dataset_temp.txt', 'a') as f:
                        f.write('Image '+str(idx+1)+'/'+str(len(self.data_list))+'\n')
            
                name, label, _ = self.data_list[idx].split(' ')
                path = os.path.join(self.data_root, name)

                img = Image.open(path)

                if self.transform:
                    img = self.transform(img)
                    
                self.images.append(img)
                self.labels.append(int(label))
                
            os.remove('dataset_temp.txt')

    def __len__(self):
        """Get length of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        if self.pre_load_files == 0:
            name, label, _ = self.data_list[idx].split(' ')

            path = os.path.join(self.data_root, name)

            img = Image.open(path)

            if self.transform:
                img = self.transform(img)
        else:
            img = self.images[idx]
            label = self.labels[idx]

        return img, int(label)


class GTSRB(Dataset):
    """GTRSB dataset."""

    def __init__(self, pre_load_files, image_dir, transform=None):
        """Init dataset.

        Keyword Parameters:
            image-dir -- directory containing images and csv annotation file
            trainsform (optional) -- transform images when loading
        """
        self.image_dir = image_dir
        self.transform = transform
        self.pre_load_files = pre_load_files

        annotations = glob.glob(image_dir + '/*.csv')

        with open(annotations[0], 'r') as f:
            self.data_list = f.readlines()[1:]
        
        if self.pre_load_files == 1:
            self.images = []
            self.labels = []
            for idx in range(len(self.data_list)):
                if (idx % 100 == 0):
                    with open('dataset_temp.txt', 'a') as f:
                        f.write('Image '+str(idx+1)+'/'+str(len(self.data_list))+'\n')
            
                name, w, h, x1, y1, x2, y2, c = self.data_list[idx].split(';')

                path = os.path.join(self.image_dir, name)
                img = Image.open(path)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Remove borders
                img = img.crop((x1, y1, x2, y2))

                if self.transform:
                    img = self.transform(img)
                    
                self.images.append(img)
                self.labels.append(int(c))
                
            os.remove('dataset_temp.txt')

    def __len__(self):
        """Lenght of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        if self.pre_load_files == 0:
            name, w, h, x1, y1, x2, y2, c = self.data_list[idx].split(';')

            path = os.path.join(self.image_dir, name)
            img = Image.open(path)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Remove borders
            img = img.crop((x1, y1, x2, y2))

            if self.transform:
                img = self.transform(img)
        else:
            img = self.images[idx]
            c = self.labels[idx]


        return img, int(c)

class Office(Dataset):
    """Office Dataset."""

    def __init__(self, pre_load_files, data_root, dataset, transform=None):
        """Initialize an Office dataset.

        Keyword Parameters:
            data_root -- root of the office dataset
            dataset -- name of the dataset to use,
                       one of ['amazon', 'dslr', 'webcam']
            trainsform (optional) -- transform images when loading
        """
        # Validate dataset parameter
        available_datasets = ['amazon', 'dslr', 'webcam']
        if dataset.lower() not in available_datasets:
            raise ValueError(f'Unknown dataset: {dataset}.'
                             f'Chose one of: {available_datasets}')

        # Get dataset directory
        dataset_dir = os.path.join(data_root, dataset, 'images')

        # Images path
        img_regexp = os.path.join(dataset_dir, '*/*')
        self.images = glob.glob(img_regexp)

        # Get labels
        classes = sorted(os.listdir(dataset_dir))
        # use // instead of \ for windows
        self.labels = list(map(lambda img: classes.index(img.split('/')[-2]),
                               self.images))

        self.transform = transform
        self.pre_load_files = pre_load_files
        
        if self.pre_load_files == 1:
            self.images_load = []
            self.labels_load = []
            for idx in range(len(self.images)):
                if (idx % 100 == 0):
                    with open('dataset_temp.txt', 'a') as f:
                        f.write('Image '+str(idx+1)+'/'+str(len(self.images))+'\n')
            
                img_name, label = self.images[idx], self.labels[idx]

                img = cv2.imread(img_name)
                img = cv2.resize(img, (256, 256))
                    
                if self.transform:
                    img = self.transform(img)
                    
                self.images_load.append(img)
                self.labels_load.append(label)
                
            os.remove('dataset_temp.txt')

    def __len__(self):
        """Get length of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get item."""
        if self.pre_load_files == 0:
            img_name, label = self.images[idx], self.labels[idx]

            img = cv2.imread(img_name)
            img = cv2.resize(img, (256, 256))

            if self.transform:
                img = self.transform(img)
        else:
            img = self.images_load[idx]
            label = self.labels_load[idx]


        return img, label