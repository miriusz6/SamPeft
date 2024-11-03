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


def adjust_lr(losses, rates):
    # if improvement is less than 0.1% stop training
    # 0.30 , 0.25
    l3 = losses[-1]
    l2 = losses[-2]
    l1 = losses[-3]
    d1 = l2 - l1
    d2 = l3 - l2
    mean_diff = (d1 + d2) / 2
    #print('mean_diff:',mean_diff)
    x = abs(mean_diff)

    last_loss = losses[-1]
    curr_rate = rates[-1]
    
    regress = mean_diff > 0


    is_lvl1 = (regress or  x < 0.01  ) and ( last_loss > 0.5 )
    is_lvl2 = (regress or  x < 0.005  ) and ( ( (last_loss < 0.5) and  (last_loss > 0.4 ) ) )
    is_lvl3 = (regress or  x < 0.003  ) and ( ( (last_loss < 0.4) and  (last_loss > 0.35) ) )
    is_lvl4 = (regress or  x < 0.002  ) and ( ( (last_loss < 0.35)  ) )

    round_X = round(x,4)

    if is_lvl1:
        print("Level 1 adjustment: ",round_X, ' < ',0.01)
        if curr_rate > 0.0005:
            return curr_rate * 0.5
        else:
            return 0.01
    elif is_lvl2:
        print("Level 2 adjustment: ",round_X, ' < ',0.005)
        if curr_rate > 0.0001:
            return curr_rate * 0.25
        else:
            return 0.005
    elif is_lvl3:
        print("Level 3 adjustment: ",round_X, ' < ',0.003)
        if curr_rate > 0.00005:
            return curr_rate * 0.1
        else:
            return 0.001
    elif is_lvl4:
        print("Level 4 adjustment: ",round_X, ' < ',0.002)
        if curr_rate > 0.00001:
            return curr_rate * 0.1
        else:
            return 0.0005
    else:
        print("No adjustment: ",round_X)
        return curr_rate

def visualize_prediction(predicted, truth):
    predicted = np.where(predicted > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)
    ret = np.zeros((256,256,3))
    # color white for truth 
    ret[truth == 1] = [255,255,255]
    # color red for false positives
    ret[(predicted == 1) & (truth == 0)] = [255,0,0]
    # color blue for false negatives
    #ret[(predicted == 0) and (truth == 1)] = [0,0,255]
    # color green for true positives
    ret[(predicted == 1) & (truth == 1)] = [0,255,0]

    return ret.astype(np.uint8)
        







#test_D = load_data(test_d=False)


# with open('train_data.pkl', 'wb') as file:
#     pickle.dump(test_D, file)


def test_model(sam,sammy, test_data: EyeData, batch_size=1):

    # with open('test_data.pkl', 'rb') as file:
    #     test_data = pickle.load(file)

    # test_data = EyeData(test_data)

    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss()
    mask_downscale_f = Upsample(scale_factor=0.5)    

  
    sam.eval()

    test_data.divide_into_batches(batch_size, augment=False)

    with torch.no_grad():
        mean_dce = 0
        mean_ce = 0
        mean_both = 0
        for batch in test_data.batches:
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
  
            mean_dce += loss_dice.item()
            mean_ce += loss_ce.item()
            mean_both += (loss_dice + loss_ce).item()

        mean_ce /= len(test_data.batches)
        mean_dce /= len(test_data.batches)
        mean_both /= len(test_data.batches)

        print(f'Eval Loss: {mean_both}, DSC: {mean_ce}, CE: {mean_dce}')

    return mean_dce, mean_ce, mean_both