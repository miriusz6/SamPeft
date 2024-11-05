import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button



# # from train_for_real import *
# # import pickle
# # from model_settings import get_args
# # from finetuneSAM.models.sam_LoRa import LoRA_Sam

# # # # model setting
# # # setting = 6

# # # args =  get_args(setting)
# # # sam = load_model(args)

# # # if args.finetune_type == 'lora':
# # #     sam = LoRA_Sam(args,sam,r=2).sam
    

# # # # checkpoints\eLdL\+checkpoint_best.pth
# # # with open('checkpoints\eLdL\+checkpoint_best.pth', 'rb') as file:
# # #     model_checkpoint = torch.load(file)

# # # sam.load_state_dict(model_checkpoint)
# # # sam.to('cuda')
# # # sammy = Sammy(sam, (512,512))


# # # with open('small_data_1.pkl', 'rb') as file:
# # #     small_data_15 = pickle.load(file)
# # # small_data_15 = EyeData(small_data_15)
# # # small_data_15.divide_into_batches(1,augment=False)
# # # mini_b = small_data_15.batches[0]


# # # imgs,ps,lbls, masks = mini_b['image'],mini_b['points'],mini_b['p_labels'],mini_b['mask']

# # # current_visual = None



# # # def revisualize():
# # #     ret = sammy.predict_w_score(input_images= imgs, points= ps, labels=lbls, masks=masks, visualize=True)
# # #     dice, bcee, iou = ret['loss_dice'], ret['loss_bce'], ret['iou']
# # #     print('Dice:',dice)
# # #     print('BCE:',bcee)
# # #     print('IoU:',iou)
# # #     global current_visual
# # #     current_visual = ret['visual'][1]






   

def main():
    #plt.axis([-20,20,0,10000])
    fig, axs = plt.subplots(nrows=3,ncols=1, width_ratios=[1],height_ratios=[0.8,0.1,0.1] ,figsize=(7, 4))
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    def on_image_click(event):
        if not event.inaxes == ax1:
            return
        print('The Click Event Triggered!:',event.xdata, event.ydata)
    cid = fig.canvas.mpl_connect('button_press_event', on_image_click)
    #check_box = CheckButtons(axs[1], ['Target','Background'], [False,False], check_props={'color':'red', 'linewidth':5})
    check_box1 = Button(ax2, 'Background', color='white', hovercolor='gold')
    check_box2 = Button(ax3, 'Target', color='white', hovercolor='gold')

    def on_checkbox_click(event):
        if event.inaxes == ax2:
            #'Background'
            check_box1.color = 'red'
            check_box2.color = 'white'
        elif event.inaxes == ax3:
            # Target
            check_box2.color = 'green'
            check_box1.color = 'white'
        else:
            return
        
    
    check_box1.on_clicked(on_checkbox_click)
    check_box2.on_clicked(on_checkbox_click)
    
    # def on_checkbox_click(label):
    #     print('The Checkbox 1 Event Triggered! ',label)
    #     if label == 'Background':
    #         check_box.set_active(0, False)
    #     # elif label == 'Target':
    #     #     check_box.set_active(1, False)
    #     else:
    #         print('Error: Unknown Label')

    
    

    axs[0].text(0.1, 0.5, 'Click me anywhere on this plot!', dict(size=20))
    plt.ion()
    plt.show()

    while True:
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()