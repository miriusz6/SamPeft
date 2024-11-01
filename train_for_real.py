from skimage.measure import label

#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
#from torchvision import datasets
from tensorboardX import SummaryWriter
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
#from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
#from finetuneSAM.utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
#from finetuneSAM.utils.losses import DiceLoss
#from finetuneSAM.utils.dsc import dice_coeff_multi_class
import cv2
import monai
#from finetuneSAM.utils.utils import vis_image


# General
import numpy as np
import torch

from skimage.measure import label

#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch

import torchvision


import matplotlib.pyplot as plt

from PIL import Image


import matplotlib.pyplot as plt


from finetuneSAM.models.sam.utils.transforms import ResizeLongestSide
from finetuneSAM.models.sam import SamPredictor, sam_model_registry
from finetuneSAM.cfg import parse_args

from mmutils import put_marks, put_mask

from os import listdir
from os.path import isfile, join
from os import getcwd
from generate_target_ps import choose_bg_points, choose_target_points



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):

    # #Fine-Tune Sam
    # args =  parse_args()


    # # setting if_mask_decoder_adapter = True puts adapters inside 2-way transformer blocks
    # # this does not change the number of decoder 2-way transformer blocks (def = 2)
    # # decoder_adapt_depth denotes how many of the two 2-way transformer blocks are adapted


    # # setting if_encoder_adapter = True puts adapters inside TinyViTBlocks in the encoder
    # # this does not change the number of encoder TinyViTBlocks (def = 4)
    # # encoder_adapt_depth (e.g. [1,2]) denotes how deep blocks will be adapted

    # args.finetune_type = "vanilla"
    # #args.finetune_type = "adapter"# "vanilla"
    # #args.if_mask_decoder_adapter = True
    # #args.image_size = 512
    # #args.decoder_adapt_depth = 1
    # args.num_cls = 2


    # Load the pre-trained model
    sam_checkpoint = "mobile_sam.pt"
    # Define the model type: Tiny Vit
    model_type = "vit_t"

    # Load the model
    mobile_sam_f = sam_model_registry[model_type](args , checkpoint=sam_checkpoint)
    # Move the model to the device
    mobile_sam_f = mobile_sam_f.to(device=device)
    # Set the model to evaluation mode
    #mobile_sam_f.eval()

    #predictor_f = SamPredictor(mobile_sam_f)
    return mobile_sam_f


def load_data():
    train_img_path = '/data/Data/train/image/'
    train_mask_path = '/data/Data/train/mask/'
    train_data = _load_data(train_img_path, train_mask_path)
    
    return train_data    
    # val_img_path = '/data/Data/val/image/'
    # val_mask_path = '/data/Data/val/mask/'
    # val_data = _load_data(val_img_path, val_mask_path)
    
    # return {'train': train_data, 'val': val_data}

def _load_data(img_path, mask_path):
    curr_dir = getcwd()
    # images
    imgs_path = curr_dir+ img_path#'/data/Data/train/image/'
    imgs_names = [f for f in listdir(imgs_path) if isfile(join(imgs_path, f))]
    # masks
    masks_path = curr_dir+ mask_path#'/data/Data/train/mask/'
    msks_names = [f for f in listdir(masks_path) if isfile(join(masks_path, f))]

    data = {'image': [], 'mask': [], 't_points': [], 'bg_points': []}



    for img_name in imgs_names:
        img = Image.open(imgs_path+img_name)
        img = np.asarray(img)
        data['image'].append(img)
    
    ts = 5
    bgs = 5

    for msk_name in msks_names:
        msk = Image.open(masks_path+msk_name)
        msk = np.asarray(msk)
        msk = np.where(msk>0, 1, 0)
        #t_points = choose_target_points(msk, ts, min_dist=50)
        #bg_points = choose_bg_points(msk, bgs, min_dist=50)
        #make dummy points
        t_points = np.array([[70,235],[218,92],[154,360]])
        bg_points = np.array([[259,257],[192,148],[220,435]])
        data['t_points'].append(t_points)
        data['bg_points'].append(bg_points)
        data['mask'].append(msk)
    return data


class Sammy:
    
    def __init__(self, model):
        self.model = model
        self.input_image = None
        self.transform = None

    def set_image(self,input_img):
        self.transform = ResizeLongestSide(input_img.shape[0])
        self.input_image = self.transform.apply_image(input_img)
        

    def encode_img(self):
        #set_image
        transformed_img = self.transform.apply_image(self.input_image)
        transformed_img = torch.as_tensor(transformed_img, device=device)
        transformed_img = transformed_img.permute(2, 0, 1).contiguous()[None, :, :, :]
        transformed_img = self.model.preprocess(transformed_img)
        features = self.model.image_encoder(transformed_img)
        return features

    def encode_promts(self, t_points, bg_points):
        labels = [1]*len(t_points) + [0]*len(bg_points)
        points = np.zeros((len(t_points)+len(bg_points),2))
        points[:len(t_points)] = t_points
        points[len(t_points):] = bg_points
        point_coords = self.transform.apply_coords(points,self.input_image.shape[:2])
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]


        points=(coords_torch, labels_torch)

        box_torch, mask_input_torch = None, None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=box_torch,
                masks=mask_input_torch,
            )

        return sparse_embeddings, dense_embeddings

    def decode_features(self, features, sparse_embeddings, dense_embeddings ):
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = features,
            image_pe= self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output= True,
        )

        # Upscale the masks to the original image resolution
        masks =self.model.postprocess_masks(low_res_masks, (256,256), (512,512))


        return masks, iou_predictions, low_res_masks

#def predict(model, input_img_size, image: np.ndarray, point = None):

    # torch.no_grad()
    # org_shape = image.shape

    # transform = ResizeLongestSide(input_img_size) # can be changed?

    # #set_image
    # input_image = transform.apply_image(image)
    # input_image_torch = torch.as_tensor(input_image, device=device)
    # input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # #set_torch_image
    # transformed_image = input_image_torch
    # input_size = tuple(transformed_image.shape[-2:])
    # transformed_image = model.preprocess(transformed_image)
    # features = model.image_encoder(transformed_image)


    # coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

    # # TO DELETE
    # t1 = np.array([70,235])
    # t2 = np.array([218,92])
    # t3 = np.array([154,360])
    # ts = np.array([t1,t2,t3])
    # # # background
    # b1 = np.array([259,257])
    # b2 = np.array([192,148])
    # b3 = np.array([220,435])
    # bs = np.array([b1,b2,b3])
    # ps = np.array([t1,t2,t3,b1,b2,b3])

    # labels = np.array([1,1,1,0,0,0])
    


    # point_coords = transform.apply_coords(ps, org_shape[:2])
    # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
    # labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
    # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]


    # points=(coords_torch, labels_torch)
    # sparse_embeddings, dense_embeddings = model.prompt_encoder(
    #         points=points,
    #         boxes=box_torch,
    #         masks=mask_input_torch,
    #     )

    # Predict masks
    # low_res_masks, iou_predictions = model.mask_decoder(
    #     image_embeddings = features,
    #     image_pe= model.prompt_encoder.get_dense_pe(),
    #     sparse_prompt_embeddings=sparse_embeddings,
    #     dense_prompt_embeddings=dense_embeddings,
    #     multimask_output= True,
    # )

    # # Upscale the masks to the original image resolution
    # masks = model.postprocess_masks(low_res_masks, input_size, org_shape[:2])


    # return masks, iou_predictions, low_res_masks



