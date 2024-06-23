import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import myDataset
from model_fusion import DecoupleFusioner


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    writer = SummaryWriter('./tensorboard/decouple')

    normalize = transforms.Normalize(mean=[0.556],
                                 std=[0.063])
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    train_set = myDataset(root='../Data/temp', transform_x=train_tfm, flag='train', fold=1)
    valid_set = myDataset(root='../Data/temp', transform_x=test_tfm, flag='valid', fold=1)

    batch_size = 16
    nw = 1
    train_num = len(train_set)
    val_num = len(valid_set)
    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(valid_set,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    net = DecoupleFusioner()
    # load pretrain weights
    fusiontransformer_path = "./fusiontransformer/fusiontransformer.pth"
    tabtransformer_path = "./tabtransformer/tabtransformer.pth"

    print("Load FusionTransformer Model.")
    assert os.path.exists(fusiontransformer_path), "file {} does not exist.".format(fusiontransformer_path)
    net.img_extractor.load_state_dict(torch.load(fusiontransformer_path, map_location='cpu'))

    print("Load TabTransformer Model.")
    assert os.path.exists(tabtransformer_path), "file {} does not exist.".format(tabtransformer_path)
    net.cli_extractor.load_state_dict(torch.load(tabtransformer_path, map_location='cpu'))

    # for param in net.img_extractor.parameters():
    #     param.requires_grad = False
    # for param in net.cli_extractor.parameters():
    #     param.requires_grad = False

    net.to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    epochs = 50
    best_acc = 0.0
    best_loss = 1.0
    save_path = './model/acc.pth'
    save_path2 = './model/loss.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    class_weight = torch.FloatTensor([1, 1]).to(device)
    
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        running_recon_loss1 = 0.0
        running_recon_loss2 = 0.0
        
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            image1, image2, image3, cli, labels = data
            optimizer.zero_grad()
            out,img_feature_rec,cli_feature_rec,img_feature,cli_feature = net(image1.to(device),image2.to(device),image3.to(device),cli.to(device))
            
            ce_loss = F.cross_entropy(out, labels.to(device), weight=class_weight)
            recon_loss = F.l1_loss(img_feature_rec,img_feature)+F.l1_loss(cli_feature_rec,cli_feature)
            if epoch > 5:
                loss = ce_loss + recon_loss
                optimizer = torch.optim.Adam(params, lr=1e-6)
            else:
                loss = ce_loss
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=10) 
            optimizer.step()

            running_loss += loss.item()
            running_recon_loss1 += recon_loss.item()
            # running_recon_loss2 += recon_loss2.item()
            predict_y = torch.max(out, dim=1)[1]
            running_acc += torch.eq(predict_y.detach(), labels.to(device).detach()).sum().item()
            
        scheduler.step()

        # validate
        net.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        num_correct = 0
        num_examples = 0
        num_tp = 0
        num_tn = 0
        num_ap = 0
        num_an = 0
        num_prep = 0
        num_pren = 0
        
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_image1, val_image2, val_image3, val_cli, val_labels = val_data
                out,img_feature_rec,cli_feature_rec,img_feature,cli_feature = net(val_image1.to(device),val_image2.to(device),val_image3.to(device),val_cli.to(device))
                loss = F.cross_entropy(out, val_labels.to(device), weight=class_weight)
                
                val_loss += loss.item()
                predict_y = torch.max(out, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                
                correct = sum(targetsi == predicti for targetsi, predicti in zip(val_labels.to(device), predict_y))
                tp = sum(targetsi and predicti for targetsi, predicti in zip(val_labels.to(device), predict_y)) 
                ap = sum(val_labels.to(device)) 
                prep = sum(predict_y)  
                tn = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(val_labels.to(device), predict_y))  
                an = len(val_labels.to(device)) - sum(val_labels.to(device))  
                pren = len(val_labels.to(device)) - sum(predict_y)  
                num_correct += correct 
                num_examples += len(labels)  
                num_tp += tp  
                num_tn += tn  
                num_ap += ap  
                num_an += an 
                num_prep += prep
                num_pren += pren

        running_accurate = running_acc / train_num
        val_accurate = val_acc / val_num
        print('[epoch %d|%d] train_loss: %.3f  recon_loss: %.8f train_accuracy: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, epochs, running_loss / train_steps, running_recon_loss1 / train_steps, running_accurate, val_loss / val_steps, val_accurate))

        yRecall = num_tp / (num_ap+ 1e-8)  
        yPrecision = num_tp / (num_prep+ 1e-8)
        yF1 = 2 * yRecall * yPrecision / (yRecall + yPrecision + 1e-8)
        nRecall = num_tn / (num_an + 1e-8) 
        nPrecision = num_tn / (num_pren+ 1e-8)
        nF1 = 2 * nRecall * nPrecision / (nRecall + nPrecision + 1e-8)        
        print('positive recall={:.4f},positive precision={:.4f},F1={:.4f} | negative recall={:.4f},negative precision={:.4f},F1={:.4f},'.format(
            yRecall, yPrecision, yF1, nRecall, nPrecision, nF1
        ))
        
        writer.add_scalar('Loss/train', running_loss / train_steps, epoch)
        writer.add_scalar('Acc/train', running_accurate, epoch)
        writer.add_scalar('Loss/valid', val_loss / val_steps, epoch)
        writer.add_scalar('Acc/valid', val_accurate, epoch)
        
        if val_accurate >= best_acc and yF1 >= 0.5 and nF1 >= 0.5:
            print('saving acc model')
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        if (val_loss / val_steps) <= best_loss:
            print('saving loss model')
            best_loss = val_loss / val_steps
            torch.save(net.state_dict(), save_path2)

    print('Finished Training')
    
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(3407)
    main()
