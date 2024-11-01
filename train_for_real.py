

#Scientific computing 
import numpy as np

import torch

from PIL import Image



# General
import numpy as np
import torch

from skimage.measure import label

#Scientific computing 
import numpy as np

#Pytorch packages
import torch

from PIL import Image


import matplotlib.pyplot as plt


from finetuneSAM.models.sam.utils.transforms import ResizeLongestSide
from finetuneSAM.models.sam import  sam_model_registry




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
    train_img_path = '/data/DataSmall/train/image/'
    train_mask_path = '/data/DataSmall/train/mask/'
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

    
    ts = 5
    bgs = 5
    labels = [1]*ts + [0]*bgs

    data = {'image': [], 'mask': [], 'points': [], 'p_labels': []}

    for img_name in imgs_names:
        img = Image.open(imgs_path+img_name)
        img = np.asarray(img)
        data['image'].append(img)
    

    for msk_name in msks_names:
        msk = Image.open(masks_path+msk_name)
        msk = np.asarray(msk)
        msk = np.where(msk>0, 1, 0)
        t_points = choose_target_points(msk, ts, min_dist=50)
        bg_points = choose_bg_points(msk, bgs, min_dist=50)
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
        transformed_imgs = torch.as_tensor(input_images, device=device)
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
            multimask_output= True,
        )

        # # Upscale the masks to the original image resolution
        # masks =self.model.postprocess_masks(low_res_masks, (1024,1024), (1024,1024))


        return iou_predictions, low_res_masks
        #return None, None, None

