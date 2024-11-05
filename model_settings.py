from finetuneSAM.cfg import parse_args

#setting if_mask_decoder_adapter = True puts adapters inside 2-way transformer blocks
# this does not change the number of decoder 2-way transformer blocks (def = 2)
# decoder_adapt_depth denotes how many of the two 2-way transformer blocks are adapted


# setting if_encoder_adapter = True puts adapters inside TinyViTBlocks in the encoder
# this does not change the number of encoder TinyViTBlocks (def = 4)
# encoder_adapt_depth (e.g. [1,2]) denotes how deep blocks will be adapted

#Fine-Tune Sam
def get_args(setting_num):
    args =  parse_args()
    args.num_cls = 3
    if setting_num == 0:
        #encoder None, decoder adapter
        args.finetune_type = "vanilla"
    if setting_num == 1:
        #encoder None, decoder adapter
        args.finetune_type = "adapter"
        args.if_mask_decoder_adapter = True
    elif setting_num == 2:
        #encoder adapter, decoder None
        args.finetune_type = "adapter"
        args.if_encoder_adapter = True
        args.if_update_encoder = True
    elif setting_num == 3:
        #encoder adapter decoder adapter
        args.finetune_type = "adapter"
        args.if_encoder_adapter = True
        args.if_mask_decoder_adapter = True
        args.if_update_encoder = True
    elif setting_num == 4:
        #encoder None, decoder lora
        args.finetune_type = "lora"
        args.if_decoder_lora_layer = True
    elif setting_num == 5:
        #encoder lora, decoder None
        args.finetune_type = "lora"
        args.if_encoder_lora_layer = True
        args.if_update_encoder = True
    elif setting_num == 6:
        #encoder lora decoder lora
        args.finetune_type = "lora"
        args.if_update_encoder = True
        args.if_encoder_lora_layer = True
        args.if_decoder_lora_layer = True
    return args




    # # # #encoder None, decoder adapter
    # # # args.finetune_type = "adapter"
    # # # args.if_mask_decoder_adapter = True


    # # #encoder adapter, decoder None      
    # # args.finetune_type = "adapter"
    # # args.if_encoder_adapter = True
    # # args.if_update_encoder = True

    # # # encoder adapter decoder adapter
    # # # args.finetune_type = "adapter"
    # # # args.if_encoder_adapter = True
    # # # args.if_mask_decoder_adapter = True
    # # # args.if_update_encoder = True



    # # # encoder None, decoder lora
    # # # args.finetune_type = "lora"
    # # # args.if_decoder_lora_layer = True


    # # # encoder lora, decoder None
    # # # args.finetune_type = "lora"
    # # # args.if_encoder_lora_layer = True

    # # # encoder lora decoder lora
    # # # args.finetune_type = "lora"
    # # # args.if_encoder_lora_layer = True
    # # # args.if_decoder_lora_layer = True