from train_for_real import *
import pickle


from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from tqdm import tqdm
import monai

from finetuneSAM.models.sam_LoRa import LoRA_Sam


from shit import test_model, adjust_lr
from finetuneSAM.cfg import parse_args
#Fine-Tune Sam
args =  parse_args()


# setting if_mask_decoder_adapter = True puts adapters inside 2-way transformer blocks
# this does not change the number of decoder 2-way transformer blocks (def = 2)
# decoder_adapt_depth denotes how many of the two 2-way transformer blocks are adapted


# setting if_encoder_adapter = True puts adapters inside TinyViTBlocks in the encoder
# this does not change the number of encoder TinyViTBlocks (def = 4)
# encoder_adapt_depth (e.g. [1,2]) denotes how deep blocks will be adapted


# parser.add_argument('-if_update_encoder', type=bool, default=False , help='if update_image_encoder')
# parser.add_argument('-if_encoder_adapter', type=bool, default=False , help='if add adapter to encoder')

# parser.add_argument('-encoder-adapter-depths', type=list, default=[0,1,10,11] , help='the depth of blocks to add adapter')
# parser.add_argument('-if_mask_decoder_adapter', type=bool, default=False , help='if add adapter to mask decoder')
# parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='the depth of the decoder adapter')

# parser.add_argument('-if_encoder_lora_layer', type=bool, default=False , help='if add lora to encoder')
# parser.add_argument('-if_decoder_lora_layer', type=bool, default=False , help='if add lora to decoder')
# parser.add_argument('-encoder_lora_layer', type=list, default=[0,1,10,11] , help='the depth of blocks to add lora, if [], it will add at each layer')

#    parser.add_argument('-finetune_type', type=str, default='adapter', help='normalization type, pick among vanilla,adapter,lora')

#args.finetune_type = "adapter"


#args.finetune_type = "vanilla"
#args.finetune_type = "adapter"# "vanilla"
#args.if_mask_decoder_adapter = True


# #encoder None, decoder adapter
# args.finetune_type = "adapter"
# args.if_mask_decoder_adapter = True


#encoder adapter, decoder None      
args.finetune_type = "adapter"
args.if_encoder_adapter = True
args.if_update_encoder = True

# encoder adapter decoder adapter
# args.finetune_type = "adapter"
# args.if_encoder_adapter = True
# args.if_mask_decoder_adapter = True
# args.if_update_encoder = True


# encoder None, decoder lora
# args.finetune_type = "lora"
# args.if_decoder_lora_layer = True


# encoder lora, decoder None
# args.finetune_type = "lora"
# args.if_encoder_lora_layer = True

# encoder lora decoder lora
# args.finetune_type = "lora"
# args.if_encoder_lora_layer = True
# args.if_decoder_lora_layer = True

is_cuda = torch.cuda.is_available()
print('Cuda:',is_cuda)

args.num_cls = 3
checkpoints_path = 'checkpoints'


sam = load_model(args)

# DATA LOADING
with open('train_data.pkl', 'rb') as file:
    data = pickle.load(file)
all_data_points = len(data['image'])


# VALIDATION DATA
val_percentage = 0.1
val_points = int(all_data_points*val_percentage)
training_points = all_data_points - val_points

val_data = {"image":[],"mask":[],"points":[],"p_labels":[]}
for i in range(val_points):
    rnd_indx = np.random.randint(0,len(data["image"])-1)
    val_data["image"].append(data["image"].pop(rnd_indx))
    val_data["mask"].append(data["mask"].pop(rnd_indx))
    val_data['points'].append(data['points'].pop(rnd_indx))
    val_data['p_labels'].append(data['p_labels'].pop(rnd_indx))
val_data = EyeData(val_data)
    
# TRAIN DATA
train_data = data
print('Train points:',training_points)
train_data = EyeData(train_data)



if args.finetune_type == 'adapter':
        for n, value in sam.named_parameters():
            if "Adapter" not in n: # only update parameters in adapter
                value.requires_grad = False
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder adapter:',args.if_encoder_adapter)
        print('if mask decoder adapter:',args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print('added adapter layers:',args.encoder_adapter_depths)
elif args.finetune_type == 'lora':
    print('if update encoder:',args.if_update_encoder)
    print('if image encoder lora:',args.if_encoder_lora_layer)
    print('if mask decoder lora:',args.if_decoder_lora_layer)
    for n, value in sam.named_parameters():
        value.requires_grad = False
    sam = LoRA_Sam(args,sam,r=4).sam



sam.to('cuda')
sammy = Sammy(sam, (512,512))


# TRAINING LOOP
b_lr = 0.01
epochs = 32
batch_size = 8

# Optimizer
optimizer = optim.AdamW(sam.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
optimizer.zero_grad()

# Metrics
criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
criterion2 = nn.BCEWithLogitsLoss()

# Downscale mask
mask_downscale_f = Upsample(scale_factor=0.5)

# Tensorboard
writer = SummaryWriter(checkpoints_path + '/log')


iter_num = 0
val_largest_dsc = 0
last_update_epoch = 0




train_data.divide_into_batches(batch_size)
print("Number of batches: ",len(train_data.batches))
print("Elems pr batch: ",len(train_data.batches[0]))
lrs = []
lss = []
pbar = tqdm(range(epochs))

#### TRAINING
print('Start training')
for epoch in pbar:
    print('Epoch num {}'.format(epoch))
    sam.train()
    train_loss = 0
    # divide before new epoch
    train_data.divide_into_batches(batch_size, augment=True)
    b_bar = tqdm(range(len(train_data.batches)))
    for i in b_bar:
        # unpack batch
        batch = train_data.batches[i]
        imgs = batch['image']
        msks = batch['mask']
        points = batch['points']
        labels = batch['p_labels']

        # ENCODE IMAGE
        if args.if_update_encoder:
            img_emb = sammy.encode_img(imgs)
        else:
            with torch.no_grad():
                img_emb = sammy.encode_img(imgs)
        
        # ENCODE PROMTS
        sparse_emb, dense_emb = sammy.encode_promts(points=points, labels=labels)

        # DECODE FEATURES
        _,pred = sammy.decode_features(img_emb, sparse_emb, dense_emb)

        # PREPARE MASKS
        msks = torch.tensor(msks).float().cuda()
        # from Bx512x512 to Bx1x512x512
        msks = msks.unsqueeze(1)
        # from 512x512 to 256x256
        msks = mask_downscale_f(msks)
        
        # LOSS
        loss_dice =  criterion1(pred,msks)
        loss_ce = criterion2(pred,msks)
        loss =  loss_dice + loss_ce

        # BACKPROP
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # METRICS
        train_loss += loss.item()
        iter_num+=1

        # TENSORBOARD
        writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/loss_ce', loss_ce, iter_num)
        writer.add_scalar('info/loss_dice', loss_dice, iter_num)
    
    train_loss /= (i+1)
    #pbar.set_description('Epoch num {}| train loss {} | lr_opt {} \n'.format(epoch,round(train_loss, 4),optimizer.param_groups[0]['lr']))
    
    print('Epoch num {}| train loss {} | lr_opt {} \n'.format(epoch,round(train_loss, 4),optimizer.param_groups[0]['lr']))

    # ADJUST LEARNING RATE
    lss.append(train_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    if len(lss) > 3:
        b_lr = adjust_lr(lss, lrs)
        #print("adjust_lr:",b_lr)
        if b_lr != optimizer.param_groups[0]['lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = b_lr
            lss = []
            lrs = []
        else:
            lss.pop(0)
            lrs.pop(0)

    ##### VALIDATION
    if epoch > 5 and epoch%5==0:
        eval_loss=0
        dsc = 0
        sam.eval()
        with torch.no_grad():
            # divide before new epoch, no augment
            val_data.divide_into_batches(1, augment=False) 
            for i in range(len(val_data.batches)):
                batch = val_data.batches[i]
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
                
                eval_loss +=loss.item()
                dsc_batch = 1-loss_dice
                dsc+=dsc_batch
                #print(dsc_batch)

            eval_loss /= (i+1)
            dsc /= (i+1)
            
            writer.add_scalar('eval/loss', eval_loss, epoch)
            writer.add_scalar('eval/dice', dsc, epoch)
            
            print('Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,round(eval_loss, 2),dsc))
            if dsc>val_largest_dsc:
                val_largest_dsc = dsc
                last_update_epoch = epoch
                print('largest DSC now: {}'.format(dsc))
                torch.save(sam.state_dict(),checkpoints_path + '/checkpoint_best.pth')
            elif (epoch-last_update_epoch)>20:
                # the network haven't been updated for 20 epochs
                print('Training finished###########')
                break
writer.close()


# TESTING
test_model(sam,sammy)