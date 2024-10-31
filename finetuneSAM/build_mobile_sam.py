from models.sam.build_sam import build_sam_vit_t

from cfg import parse_args
import numpy as np
import torchvision as to

# parser.add_argument('-sam_ckpt', type=str, default='sam_vit_b_01ec64.pth', help='the path to the checkpoint to load')
#m = build_sam_vit_t(parse_args())


args = parse_args()
args.sam_ckpt = 'mobile_sam_weights.pt'
print(args.sam_ckpt)

m = build_sam_vit_t(parse_args())

img = to.io.read_image("datasets/MRI-Prostate/images/patient56/study56/volumn56/I2CVB01_27.png") 

m.preprocess(img)

print(type(img))

m(img,multimask_output=True)