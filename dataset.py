import torch.utils.data as data
from PIL import Image
import os
import os.path
import pickle
import torch

def split(data, flag):
    train = []
    valid = []
    test = []
    n = 0
    for i in data:
        if n == 8:
            valid.append(i)
            n = n+1
            #n = 0
        elif n == 7:
            test.append(i)
            n = n+1
        else:
            train.append(i)
            if n == 9:
                n = 0
            else:
                n = n+1

    if flag == 'train':
        return train

    if flag == 'valid':
        return valid

    if flag == 'test':
        return test


def make_dataset(dir, flag, catagory):
    
    imagesPath = []
    labelsPath = []
    clinicalsPath = []
    
    if catagory == 'single':
        with open(os.path.join(dir, 'single22_img.pkl'), 'rb') as f:
            imagesPath = pickle.load(f)
            imagesPath = split(imagesPath, flag)
        with open(os.path.join(dir, 'single22_label.pkl'), 'rb') as f:
            labelsPath = pickle.load(f)
            labelsPath = split(labelsPath, flag)
        with open(os.path.join(dir, 'single22_clinical.pkl'), 'rb') as f:
            clinicalsPath = pickle.load(f)
            clinicalsPath = split(clinicalsPath, flag)
    
    if catagory == 'double':
        with open(os.path.join(dir, 'double22_img.pkl'), 'rb') as f:
            imagesPathDouble = pickle.load(f)
            imagesPathDouble = split(imagesPathDouble, flag)
        with open(os.path.join(dir, 'double22_label.pkl'), 'rb') as f:
            labelsPathDouble = pickle.load(f)
            labelsPathDouble = split(labelsPathDouble, flag)
        with open(os.path.join(dir, 'double22_clinical.pkl'), 'rb') as f:
            clinicalsPathDouble = pickle.load(f)
            clinicalsPathDouble = split(clinicalsPathDouble, flag)
        if flag == 'train':
            for i in range(len(imagesPathDouble)):
                if int(labelsPathDouble[i]) == 1:
                    for j in range(1, 3):
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
                else:
                    for j in range(1, 3):
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
        else:
            for i in range(len(imagesPathDouble)):
                for j in range(1, 3):
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
    
    if catagory == 'both':
        with open(os.path.join(dir, 'single22_img.pkl'), 'rb') as f:
            imagesPath = pickle.load(f)
            imagesPath = split(imagesPath, flag)
        with open(os.path.join(dir, 'single22_label.pkl'), 'rb') as f:
            labelsPath = pickle.load(f)
            labelsPath = split(labelsPath, flag)
        with open(os.path.join(dir, 'single22_clinical.pkl'), 'rb') as f:
            clinicalsPath = pickle.load(f)
            clinicalsPath = split(clinicalsPath, flag)
        
        with open(os.path.join(dir, 'double22_img.pkl'), 'rb') as f:
            imagesPathDouble = pickle.load(f)
            imagesPathDouble = split(imagesPathDouble, flag)
        with open(os.path.join(dir, 'double22_label.pkl'), 'rb') as f:
            labelsPathDouble = pickle.load(f)
            labelsPathDouble = split(labelsPathDouble, flag)
        with open(os.path.join(dir, 'double22_clinical.pkl'), 'rb') as f:
            clinicalsPathDouble = pickle.load(f)
            clinicalsPathDouble = split(clinicalsPathDouble, flag)
    
        if flag == 'train':
            for i in range(len(imagesPathDouble)):
                if int(labelsPathDouble[i]) == 1:
                    for j in range(1, 3):
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
                else:
                    for j in range(1, 3):
                        imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                        labelsPath.append(labelsPathDouble[i])
                        clinicalsPath.append(clinicalsPathDouble[i])
        else:
            for i in range(len(imagesPathDouble)):
                for j in range(1, 3):
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
    
    return imagesPath, labelsPath, clinicalsPath

class myDataset(data.Dataset):

    def __init__(self, root, transform_x=None, flag='train', catagory='double'):

        image, label, clinical = make_dataset(root, flag, catagory)
        self.flag = flag
        self.transform = transform_x
        self.imagePath = image
        self.labelPath = label
        self.clinicalPath = clinical

    def __getitem__(self, index):

        name = self.imagePath[index]
        name = name.replace('\\', '/')
        label = self.labelPath[index]
        clinical = self.clinicalPath[index]
        clinical[21]=0

        cli = torch.tensor(clinical, dtype=torch.float64)
        cli = cli.type(torch.FloatTensor)
        
        image1 = Image.open(name + '-d1.jpg').convert('L')
        image1 = self.transform(image1)
        
        image2 = Image.open(name + '-d2.jpg').convert('L')
        image2 = self.transform(image2)
        
        image3 = Image.open(name + '-d3.jpg').convert('L')
        image3 = self.transform(image3)

        return image1,image2,image3,cli,int(label)

    def __len__(self):

        return len(self.imagePath)