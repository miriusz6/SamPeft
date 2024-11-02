from train_for_real import *
import pickle

# #Scientific computing 
# import numpy as np
# #Pytorch packages
# import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
#Visulization
from tqdm import tqdm
import monai



#test_D = load_data(test_d=True)

#save dta as pickle
# with open('test_data.pkl', 'wb') as file:
#     pickle.dump(test_D, file)


def test_model(sam,sammy):

    with open('test_data.pkl', 'rb') as file:
        test_data = pickle.load(file)

    test_data = EyeData(test_data)

    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss()
    mask_downscale_f = Upsample(scale_factor=0.5)    

  
    sam.eval()

    test_data.divide_into_batches(test_data.length)

    with torch.no_grad():
        batch = test_data.batches[0]
        imgs = batch['image']
        msks = batch['mask']
        points = batch['points']
        labels = batch['p_labels']

        img_emb = sammy.encode_img(imgs)
        
        sparse_emb, dense_emb = sammy.encode_promts(points=points, labels=labels)
        _,pred = sammy.decode_features(img_emb, sparse_emb, dense_emb)

        msks = torch.tensor(msks).float().cuda()
        # from Bx512x512 to Bx1x512x512
        msks = msks.unsqueeze(1)
        # from 512x512 to 256x256
        msks = mask_downscale_f(msks)
        
        loss_dice =  criterion1(pred,msks)
        loss_ce = criterion2(pred,msks)
        loss =  loss_dice + loss_ce
        
        eval_loss =loss.item()
        dsc_batch = 1-loss_dice
        dsc = dsc_batch
        #print(dsc_batch)

        
        print(f'Eval Loss: {eval_loss}, DSC: {dsc}')