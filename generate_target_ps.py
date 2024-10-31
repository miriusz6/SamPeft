import numpy as np




# function distance in 2D
def distance(p1, p2):
    return np.sqrt((p1.X - p2.X)**2 + (p1.Y - p2.Y)**2)



def choose_bg_points(img:np.array, amount = 5, min_dist = 20):

    img = preprocess_bg_mask(img)
    
    if img.sum() < amount:
        # throw exception
        raise ValueError('Amount of points is greater than the amount of ones in the mask')

    chosen = []
    while len(chosen) < amount:

        chosen = []
        ps = create_points(img)
        # sort based on density
        ps.sort(key=lambda x: x.density, reverse=False)
        chosen = [ps.pop(0)]

        def find_far_enough():
            for i in range(len(ps)):
                distances = [distance(ps[i],choice) for choice in chosen]
                if min(distances) >= min_dist:
                    return i
            return -1
        

        for i in range(amount-1):
            i = find_far_enough()
            if i == -1:
                break
            chosen.append(ps[i])
            ps = ps[i+1:]
       
        min_dist -= 5
    
    print("distance applied: ", min_dist)
    chosen = np.array([np.array([p.X,p.Y]) for p in chosen])

    return chosen

def choose_target_points(img:np.array, amount = 5, min_dist = 20):

    img = preprocess_target_mask(img)
    
    # if img.sum() < amount:
    #     # throw exception
    #     raise ValueError('Amount of points is greater than the amount of ones in the mask')

    chosen = []
    while len(chosen) < amount:

        chosen = []
        ps = create_points(img)
        # sort based on density
        ps.sort(key=lambda x: x.density, reverse=True)
        chosen = [ps.pop(0)]

        def find_far_enough():
            for i in range(len(ps)):
                distances = [distance(ps[i],choice) for choice in chosen]
                if min(distances) >= min_dist:
                    return i
            return -1
        

        for i in range(amount-1):
            i = find_far_enough()
            if i == -1:
                break
            chosen.append(ps[i])
            ps = ps[i+1:]
       
        min_dist -= 5
    
    print("distance applied: ", min_dist)
    chosen = np.array([np.array([p.X,p.Y]) for p in chosen])

    return chosen


def create_points(img):
    ones = np.where(img == 1)
    coords = list(zip(ones[0], ones[1]))
    points = []
    for x,y in coords:
        p = Point(x,y)
        p.density = calc_density(img, (x,y))
        if p.density == -1:
            continue
        points.append(p)
    return points
    

def calc_density(img, point):
    density = 0
    # sum pixels in a 5x5 square around the point
    square_size = 10
    try:
        for i in range(-square_size//2,square_size//2):
            for j in range(-square_size//2,square_size//2):
                density += img[point[0]+i][point[1]+j]
    except:
        density = -1
    return density

class Point:
    X = 0
    Y = 0
    density = 0
    def __init__(self,x,y):
        self.X = x
        self.Y = y





# plot img and points
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('data/Data/test/mask/0.png')

img = np.array(img)
org_img = img.copy()


def preprocess_target_mask(img):
    img = np.where(img > 0, 1, 0)   
    img = ndimage.binary_erosion(img, structure=np.ones((3,3)))
    img = ndimage.binary_erosion(img, structure=np.ones((2,2)))
    # reverse ones and zeross
    img = np.where(img == 1, 0, 1)
    img = ndimage.binary_dilation(img, structure=np.ones((2,2)))
    img = np.where(img == 0, 1, 0)
    img = ndimage.binary_dilation(img, structure=np.ones((3,3)))
    return img

def preprocess_bg_mask(img):
    img = np.where(img > 0, 1, 0)   
    img = ndimage.binary_dilation(img, structure=np.ones((10,10)))
    img = np.where(img == 1, 0, 1)
    return img

from scipy import ndimage


## raw_img = Image.open('data/Data/test/image/0.png')

# # pts = choose_target_points(img, 10, min_dist=75)

# # pbgs = choose_bg_points(img, 10, min_dist=75)


