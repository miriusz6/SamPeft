

def put_mask(image,mask):
    img = image.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                img[i,j] = [255,0,0]
    return img



def put_marks(img, coords, mark_size, target = False):
    img = img.copy()
    for coord in coords:
        x = int(coord[0])
        y = int(coord[1])
        if target:
            val = [0,0 , 255]
        else:
            val = [255, 0, 0]
        # put cross
        img[y-mark_size:y+mark_size, x-mark_size:x+mark_size] = val
        #img[y-mark_size:y+mark_size, x-mark_size:x+mark_size] = val
    return img


# def format_results(masks, scores, logits, filter=0):
#     annotations = []
#     n = len(scores)
#     for i in range(n):
#         annotation = {}

#         mask = masks[i]
#         tmp = np.where(mask != 0)
#         if np.sum(mask) < filter:
#             continue
#         annotation["id"] = i
#         annotation["segmentation"] = mask
#         annotation["bbox"] = [
#             np.min(tmp[0]),
#             np.min(tmp[1]),
#             np.max(tmp[1]),
#             np.max(tmp[0]),
#         ]
#         annotation["score"] = scores[i]
#         annotation["area"] = annotation["segmentation"].sum()
#         annotations.append(annotation)
#     return annotations