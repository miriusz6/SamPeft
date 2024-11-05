import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button





from train_for_real import *
import pickle
from model_settings import get_args
from finetuneSAM.models.sam_LoRa import LoRA_Sam

# model setting
setting = 6

args =  get_args(setting)
sam = load_model(args)

if args.finetune_type == 'lora':
    sam = LoRA_Sam(args,sam,r=2).sam
    

# checkpoints\eLdL\+checkpoint_best.pth
with open('checkpoints\eLdL\+checkpoint_best.pth', 'rb') as file:
    model_checkpoint = torch.load(file)

sam.load_state_dict(model_checkpoint)
sam.to('cuda')
sammy = Sammy(sam, (512,512))




with open('small_data_1.pkl', 'rb') as file:
    small_data_1 = pickle.load(file)
small_data_1 = EyeData(small_data_1)
small_data_1.divide_into_batches(1,augment=False)
mini_b = small_data_1.batches[0]


imgs,ps,lbls, masks = mini_b['image'],mini_b['points'],mini_b['p_labels'],mini_b['mask']

sammy.set_images(imgs)

ret = sammy.predict_from_set(ps,lbls,masks, visualize=True)

current_image = ret['visual'][0]

target_points = list(ps[0][:sum(lbls[0])])
background_points = list(ps[0][sum(lbls[0]):])



def revisualize(new_target_points, new_background_points):
    global current_image
    global target_points
    global background_points
    global current_visual
    prev_target_points = np.array(target_points).reshape(1,len(target_points),2)
    prev_background_points = np.array(background_points).reshape(1,len(background_points),2)
    all_points = np.zeros(
        (1,
         len(target_points)+len(background_points),
          2
        ))
    prev_t_ps = prev_target_points.shape[1]
    prev_b_ps = prev_background_points.shape[1]
    new_t_ps = new_target_points.shape[1]
    new_b_ps = new_background_points.shape[1]
    
    all_points[0][:prev_t_ps] = prev_target_points
    all_points[0][prev_t_ps:prev_t_ps+new_t_ps] = new_target_points
    all_points[0][prev_t_ps+new_t_ps:prev_t_ps+new_t_ps+prev_b_ps] = prev_background_points
    all_points[0][prev_t_ps+new_t_ps+prev_b_ps:] = new_background_points
    
    all_labels = np.zeros(
        (1,
         len(target_points)+len(background_points)
        ))
    all_labels[0][:prev_t_ps+new_t_ps] = 1
    all_labels.reshape(1,all_labels.shape[1],1)
    _ret = sammy.predict_from_set(all_points,all_labels,masks, visualize=True)
    current_visual = _ret['visual'][0]

revisualize(np.array([[1,1]]),np.array([[2,2]]))
# plot the image


def draw_points(target_points, background_points):
    # draw points 5x5
    for point in target_points:
        plt.plot(point[0],point[1],)

plt.imshow(current_image)

point_label = None
   

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
        # redraw with new points
        # change image
        
    cid = fig.canvas.mpl_connect('button_press_event', on_image_click)
    
    
    bg_button = Button(ax2, 'Background', color='white', hovercolor='gold')
    t_button = Button(ax3, 'Target', color='white', hovercolor='gold')

    def on_checkbox_click(event):
        global point_label
        if event.inaxes == ax2:
            #'Background'
            if bg_button.color == 'red':
                bg_button.color = 'white'
                point_label = None
            else:
                bg_button.color = 'red'
                point_label = 0
            t_button.color = 'white'
        elif event.inaxes == ax3:
            # Target
            if t_button.color == 'green':
                t_button.color = 'white'
                point_label = None
            else:
                t_button.color = 'green'
                point_label = 1
            bg_button.color = 'white'
        else:
            return
        
    bg_button.on_clicked(on_checkbox_click)
    t_button.on_clicked(on_checkbox_click)
    
    
    

    axs[0].text(0.1, 0.5, 'Click me anywhere on this plot!', dict(size=20))
    plt.ion()
    plt.show()

    while True:
        plt.draw()
        plt.pause(0.001)
        


if __name__ == '__main__':
    main()