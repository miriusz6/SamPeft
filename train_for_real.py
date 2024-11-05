

#Scientific computing 
import numpy as np

import torch

from PIL import Image



# General
import numpy as np
import torch


#Scientific computing 
import numpy as np

#Pytorch packages
import torch

from PIL import Image


from finetuneSAM.models.sam import  sam_model_registry

import torchvision.transforms.v2 as tr 


from os import listdir
from os.path import isfile, join
from os import getcwd
from generate_target_ps import choose_bg_points, choose_target_points
from tqdm import tqdm
import torchvision

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeData():
    def __init__(self, data):
        self.length = len(data['image'])
        self.images = data['image']
        self.masks = data['mask']
        self.points = data['points']
        self.p_labels = data['p_labels']
        self.batches = [] 
    
    def divide_into_batches(self, batch_size, augment = True):
        self.batches = []
        indxs = np.random.permutation(self.length)
        for i in range(0, self.length, batch_size):
            batch_indxs = indxs[i:i+batch_size]
            imgs = [self.images[j] for j in batch_indxs]
            imgs = np.array(imgs)
            masks = [self.masks[j] for j in batch_indxs]
            masks = np.array(masks)
            points = [self.points[j] for j in batch_indxs]
            points = np.array(points)
            p_labels = [self.p_labels[j] for j in batch_indxs]
            p_labels = np.array(p_labels)
            batch = {'image': imgs, 'mask': masks, 'points': points, 'p_labels': p_labels}
            if augment:
                batch = self.augment_batch(batch)

            # normalize images
            pixel_mean  = [123.675, 116.28, 103.53],
            pixel_std = [58.395, 57.12, 57.375]
            imgs = batch['image']
            batch['image'] = (imgs - pixel_mean) / pixel_std
            self.batches.append(batch)
            
    def augment_batch(self, batch):
        imgs = batch['image']
        masks = batch['mask']
        points = batch['points']
        p_labels = batch['p_labels']
        # get random angles
        angles = np.random.randint(0, 359, len(imgs))
        #angles = [90]*len(imgs)
        
        # rotate masks
        masks = self.rotate_imgs(masks, angles)

        # rotate images
        imgs = self.rotate_imgs(imgs, angles)

        # add random noise
        imgs = self.add_g_noise(imgs)

        # rotate points
        points = [self.rotate_points((512//2,512//2), p, math.radians(a)) for p, a in zip(points, angles)]

        

        return {'image': imgs, 'mask': masks, 'points': points, 'p_labels': p_labels}
     
    

    def rotate_p_origin(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in rad.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return np.array([int(qx), int(qy)])

    def rotate_points(self,origin, points, angle):
        return np.array([self.rotate_p_origin(origin, p, angle) for p in points])
    
    def rotate_img(self,img, angle):
        img = Image.fromarray(img)
        img = img.rotate(angle)
        return np.array(img)
    
    def rotate_imgs(self, imgs, angles):
        return np.array([self.rotate_img(img, angle) for img, angle in zip(imgs, angles)])
    
    def add_g_noise(self, imgs):
        imgs = torch.tensor(imgs).float()
        imgs = imgs.permute(0, 3, 1, 2)#.contiguous()
        t = tr.GaussianNoise()
        imgs = imgs + t(imgs)
        imgs = imgs.permute(0, 2, 3, 1)#.contiguous()
        imgs = imgs.long()
        return imgs.numpy()
        


    # make indexable
    def __getitem__(self, index):
        return {'image': self.images[index], 'mask': self.masks[index], 'points': self.points[index], 'p_labels': self.p_labels[index]}

    def __len__(self):
        return self.length
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.length:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration
        
    


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


def load_data(test_d = False):
    
    if test_d:
        train_img_path = '/data/Data/test/image/'
        train_mask_path = '/data/Data/test/mask/'
    else:
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

    
    ts = 1
    bgs = 1
    labels = [1]*ts + [0]*bgs

    data = {'image': [], 'mask': [], 'points': [], 'p_labels': []}

    for img_name in imgs_names:
        img = Image.open(imgs_path+img_name)
        img = np.asarray(img)
        data['image'].append(img)
    
    pbar = tqdm(range(len(msks_names)))
    for i in pbar:
        msk_name = msks_names[i]
        msk = Image.open(masks_path+msk_name)
        msk = np.asarray(msk)
        msk = np.where(msk>0, 1, 0)
        t_points = choose_target_points(msk, ts, min_dist=100)
        bg_points = choose_bg_points(msk, bgs, min_dist=100)
        #make dummy points
        # t_points = [(1,1)]*ts
        # bg_points = [(2,2)]*bgs
        points = np.zeros((ts+bgs,2))
        points[:ts] = t_points
        points[ts:] = bg_points
        data['points'].append(points)
        data['p_labels'].append(labels)
        data['mask'].append(msk)

    
    return data

from torch.nn import Upsample
import monai
from torch import nn
from shit import visualize_prediction
class Sammy:
    
    def __init__(self, model, orginal_input_size):
        self.model = model
        self.garbage = None


        self.input_img_scale = 1024/orginal_input_size[0]
        self.output_mask_scale = 256/orginal_input_size[0]

        self.input_img_scaleF = Upsample(scale_factor=self.input_img_scale)
        self.output_mask_scaleF = Upsample(scale_factor=self.output_mask_scale)

        
    def encode_img(self, input_images: np.ndarray):
        """
        Encodes the input images using a pre-trained model.
        Args:
            input_images (torch.Tensor): The input images to be encoded.[B,3,H,W]
        Returns:
            torch.Tensor: The encoded features of the input images.
        """
        
        #set_image
        input_images = np.array(input_images)
        transformed_imgs = torch.as_tensor(input_images, device=device, dtype=torch.float32)
        transformed_imgs = transformed_imgs.permute(0, 3, 1, 2)#.contiguous()
        transformed_imgs = self.input_img_scaleF(transformed_imgs)
        transformed_imgs = self.model.preprocess(transformed_imgs) 
        features = self.model.image_encoder(transformed_imgs)
        return features

    def encode_promts(self, points, labels):
        #point_coords = self.transform.apply_coords(points,self.input_image.shape[:2])
        points = np.array(points) * self.input_img_scale 
        coords_torch = torch.as_tensor(points, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
        #coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]


        points=(coords_torch, labels_torch)

        box_torch, mask_input_torch = None, None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=box_torch,
                masks=mask_input_torch,
            )

        return sparse_embeddings, dense_embeddings
        # labels = [1]*len(t_points) + [0]*len(bg_points)
        # points = np.zeros((len(t_points)+len(bg_points),2))
        # points[:len(t_points)] = t_points
        # points[len(t_points):] = bg_points
        # #point_coords = self.transform.apply_coords(points,self.input_image.shape[:2])
        # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        # labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)
        # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]


        # points=(coords_torch, labels_torch)

        # box_torch, mask_input_torch = None, None
        # sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
        #         points=points,
        #         boxes=box_torch,
        #         masks=mask_input_torch,
        #     )

        # return sparse_embeddings, dense_embeddings

    def decode_features(self, features, sparse_embeddings, dense_embeddings ):
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = features,
            image_pe= self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output= False,
        )

        # # Upscale the masks to the original image resolution
        # masks =self.model.postprocess_masks(low_res_masks, (1024,1024), (1024,1024))


        return iou_predictions, low_res_masks
        #return None, None, None


    def predict(self, input_images, points, labels):
        with torch.no_grad():
            img_emb = self.encode_img(input_images)
            sparse_emb, dense_emb = self.encode_promts(points=points, labels=labels)
            iou_predictions, pred = self.decode_features(img_emb, sparse_emb, dense_emb)
        return iou_predictions, pred
    
    def predict_w_score(self, input_images, points, labels, masks, visualize = False):
        with torch.no_grad():
            img_emb = self.encode_img(input_images)
            sparse_emb, dense_emb = self.encode_promts(points=points, labels=labels)
            iou_predictions, pred = self.decode_features(img_emb, sparse_emb, dense_emb)

        msks = torch.tensor(masks).float().cuda()
        # from Bx512x512 to Bx1x512x512
        msks = msks.unsqueeze(1)
        # from 512x512 to 256x256
        mask_downscale_f = Upsample(scale_factor=0.5)
        msks = mask_downscale_f(msks)
        
        criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss()
        loss_dice =  criterion1(pred,msks)
        loss_bce = criterion2(pred,msks)

        visual = None
        if visualize:
            visual = []
            for i in range(0, pred.shape[0]):
                p = pred[i][0].detach().cpu().numpy()
                m = msks[i][0].detach().cpu().numpy()
                visual.append(visualize_prediction(p,m))
            visual = np.array(visual)
        return {'iou': iou_predictions, 'pred': pred, 'loss_dice': loss_dice, 'loss_bce': loss_bce, 'visual': visual}
