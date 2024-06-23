import torch.utils.data as data
from PIL import Image
import os
import os.path
import pickle
import torch

def make_dataset(dir, flag, fold):
    
    imagesPath = []
    labelsPath = []
    clinicalsPath = []

    if fold == 0: #testset
        with open(os.path.join(dir, 'double22_img'+flag+'.pkl'), 'rb') as f:
            imagesPathDouble = pickle.load(f)
        with open(os.path.join(dir, 'double22_label'+flag+'.pkl'), 'rb') as f:
            labelsPathDouble = pickle.load(f)
        with open(os.path.join(dir, 'double22_clinical'+flag+'.pkl'), 'rb') as f:
            clinicalsPathDouble = pickle.load(f)
        for i in range(len(imagesPathDouble)):
            for j in range(1, 3):
                imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                labelsPath.append(labelsPathDouble[i])
                clinicalsPath.append(clinicalsPathDouble[i])
    else: #trainset and validset
        with open(os.path.join(dir, 'double22_img'+flag+str(fold)+'.pkl'), 'rb') as f:
            imagesPathDouble = pickle.load(f)
        with open(os.path.join(dir, 'double22_label'+flag+str(fold)+'.pkl'), 'rb') as f:
            labelsPathDouble = pickle.load(f)
        with open(os.path.join(dir, 'double22_clinical'+flag+str(fold)+'.pkl'), 'rb') as f:
            clinicalsPathDouble = pickle.load(f)
        if flag == 'train': #trainset
            for i in range(len(imagesPathDouble)):
                if int(labelsPathDouble[i]) == 1: #double positive sample
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
        else: #validset
            for i in range(len(imagesPathDouble)):
                for j in range(1, 3):
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
    
    return imagesPath, labelsPath, clinicalsPath

class myDataset(data.Dataset):

    def __init__(self, root, transform_x=None, flag='train', fold=1):

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
