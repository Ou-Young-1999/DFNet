import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from dataset import myDataset
from model_fusion import DecoupleFusioner

import pickle

def main():
    device = torch.device("cpu")
    print("using {} device.".format(device))
    
    normalize = transforms.Normalize(mean=[0.556],
                                 std=[0.063])
    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    test_set = myDataset(root='../Data/temp', transform_x=test_tfm, flag='test',fold=0)

    batch_size = 1
    nw = 1  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    test_num = len(test_set)
    print("using {} images for test.".format(test_num))
    
    net = DecoupleFusioner()
    # load pretrain weights
    model_path = "./model/decouple.pth"

    print("Load Model.")
    assert os.path.exists(model_path), "file {} does not exist.".format(model_path)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.to(device)
    
    # test
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    num_correct = 0
    num_examples = 0
    num_tp = 0
    num_tn = 0
    num_ap = 0
    num_an = 0
    num_prep = 0
    num_pren = 0
    
    pre = []
    label = []
    out = []
    
    softmax = nn.Softmax(dim=1)
    img_related_list = []
    cli_related_list = []
    img_unrelated_list = []
    cli_unrelated_list = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_image1, test_image2, test_image3, test_cli, test_labels = test_data
            fusion_out,img_feature_rec,cli_feature_rec,img_feature,cli_feature = net(test_image1.to(device),test_image2.to(device),test_image3.to(device),test_cli.to(device))
            # loss = loss_function(outputs, test_labels)
            # img_related_list.append(np.around(img_related.numpy()[0], 4))
            # cli_related_list.append(np.around(cli_related.numpy()[0], 4))
            # img_unrelated_list.append(np.around(img_unrelated.numpy()[0], 4))
            # cli_unrelated_list.append(np.around(cli_unrelated.numpy()[0], 4))
            predict_y = torch.max(fusion_out, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            
            soft = softmax(fusion_out)
            pre.append(predict_y.numpy()[0])
            label.append(test_labels.numpy()[0])
            out.append(soft.numpy()[0][1])
            
            correct = sum(targetsi == predicti for targetsi, predicti in zip(test_labels, predict_y))
            tp = sum(targetsi and predicti for targetsi, predicti in zip(test_labels, predict_y)) 
            ap = sum(test_labels) 
            prep = sum(predict_y) 
            tn = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(test_labels, predict_y)) 
            an = len(test_labels) - sum(test_labels) 
            pren = len(test_labels) - sum(predict_y) 
            num_correct += correct 
            num_examples += len(test_labels) 
            num_tp += tp 
            num_tn += tn 
            num_ap += ap 
            num_an += an 
            num_prep += prep
            num_pren += pren

    val_accurate = acc / test_num
    yRecall = num_tp / (num_ap+ 1e-8) 
    yPrecision = num_tp / (num_prep+ 1e-8)
    yF1 = 2 * yRecall * yPrecision / (yRecall + yPrecision + 1e-8)
    nRecall = num_tn / (num_an + 1e-8)
    nPrecision = num_tn / (num_pren+ 1e-8)
    nF1 = 2 * nRecall * nPrecision / (nRecall + nPrecision + 1e-8)
    print('Test Acc: %.3f' %(val_accurate))
    print('positive recall={:.3f},positive precition={:.3f},yF1={:.3f} | negative recall={:.3f},negative precision={:.3f},nF1={:.3f},'.format(
        yRecall, yPrecision, yF1, nRecall, nPrecision, nF1))  
    
    auc = roc_auc_score(label,out)
    report = classification_report(label,pre,digits=3)
    print('Test AUC: %.3f' %(auc))

    with open("./result/fusion.txt", 'w', encoding="utf-8") as f:        
        f.write('Test AUC: %.3f\n' %(auc))
        f.write(report)
        f.write('label   pre   probability'+'\n')
        for i in range(len(label)):
            f.write(str(label[i])+'   ')
            f.write(str(pre[i])+'   ')
            f.write(str(out[i])+'\n')
    '''
    with open('./result/img_related.pkl', 'wb') as f:
        pickle.dump(img_related_list, f)
    with open('./result/cli_related.pkl', 'wb') as f:
        pickle.dump(cli_related_list, f)
    
    with open('./result/img_unrelated.pkl', 'wb') as f:
        pickle.dump(img_unrelated_list, f)
    with open('./result/cli_unrelated.pkl', 'wb') as f:
        pickle.dump(cli_unrelated_list, f)
    '''
    
if __name__ == '__main__':
    main()
