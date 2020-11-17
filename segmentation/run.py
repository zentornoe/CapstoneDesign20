import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import argparse
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

from model import Unet
from preprocess import pad_resize

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    separable = args.separable
    up_method = args.up_method
    down_method = args.down_method
    img_src = args.s

    ######## Model Setting ########
    model = Unet(input_dim=1, separable=separable,
                 down_method=down_method, up_method=up_method)
    model = nn.DataParallel(model).to(device)
    load_state = torch.load(f'./checkpoint/{down_method}_{up_method}_{separable}.ckpt')
    model.load_state_dict(load_state['model_state_dict'])
    model.eval()
    ###############################

    d = dcrf.DenseCRF2D(512, 512, 2)

    img = Image.open(img_src).convert('L')
    img = pad_resize(img, 512)
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        img = img.to(device)
        pred = model(img)

        probs = torch.stack([1-pred, pred], dim=0).cpu().numpy()
        img, pred = img.cpu().numpy(), pred.cpu().numpy()
        pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=0)
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(pairwise_energy, compat=10)

        Q = d.inference(100)
        map = np.argmax(Q, axis=0).reshape((512, 512))

        img = (255. / img.max() * (img - img.min())).astype(np.uint8)
        pred = (255. / map.max() * (map - map.min())).astype(np.uint8)

        img = Image.fromarray(np.squeeze(img), mode='L')
        pred = Image.fromarray(pred, mode='L')

        img.save(f'../similarity/{img_src[:-4]}_o.png')
        pred.save(f'../similarity/{img_src[:-4]}_p.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Femur Bone Segmentation')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument(
        '--separable',
        action='store_true',
        default=True
    )
    parser.add_argument(
        '--up_method',
        type=str,
        default='transpose',
        choices=['bilinear', 'transpose']
    )
    parser.add_argument(
        '--down_method',
        type=str,
        default='conv',
        choices=['maxpool', 'conv']
    )
    parser.add_argument(
        '-s',
        default='1_0000.png',
        type=str
    )
    args = parser.parse_args()
    main(args)