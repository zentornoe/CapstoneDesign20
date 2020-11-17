# *** IMPORT ***
import cv2
import numpy as np
from functions import image_similarity
from functions import _rotate, _expand, _angle, _division, _separate, _fracWhere, _cutting, _crop_expand, _crop,plotshow2
import argparse

# INPUT : cv2_image - cv2.imread(FILE_PATH)
# OUTPUT : angles

def main(args):
    src = args.s

    tutorial = cv2.imread('tutorial_image/tutorial_image.png')
    cv2.imshow('tutorial', tutorial)
    cv2.waitKey(10)
    target_image = cv2.imread(src)

    s = input('Select the case:')
    if (s == str(1)):
        case1(target_image)
    elif (s == str(2)):
        case2(target_image)
    elif (s == str(3)):
        case3(target_image)
    else:
        print('uncorrect case')

## *** Functions for Cases ***

# Fracture X, Area division X
def case1(cv_image):
    ang = 0
    _rotate(_expand(cv_image), _angle(cv_image))  # image -> check similarity
    change=_crop_expand(_crop(_rotate(_expand(cv_image), _angle(cv_image))))
    ang=image_similarity(change,1)

    return ang

# Fracture X, Area division O


# Fracture O, Overlap X
def case2(cv_image):
    img0, img1 = _separate(cv_image)
    ang0 = _angle(img0)
    ang1 = _angle(img1)

    _rotate(_expand(img0), ang0)  # image 0 -> check similarity
    _rotate(_expand(img1), ang1)  # image 1 -> check similarity
    changed0 = _crop_expand(_crop(_rotate(_expand(img0), ang0)))
    changed1 = _crop_expand(_crop(_rotate(_expand(img1), ang1)))
    ang0 = image_similarity(changed0,2)
    ang1 = image_similarity(changed1,2)

    plotshow2(changed0,changed1,ang0,ang1)






    return ang0, ang1


# Fracture O, Overlap O
def case3(cv_image):
    image = _rotate(_expand(cv_image), _angle(cv_image))
    cv_image1 = np.copy(image)
    n0, n1 = _fracWhere(image)
    img0, img1 = _cutting(cv_image1, n0, n1)
    ang0 = _angle(img0)
    ang1 = _angle(img1)

    _rotate(_expand(img0), ang0)  # image 0 -> check similarity
    _rotate(_expand(img1), ang1)  # image 1 -> check similarity
    _crop_expand(_crop(_rotate(_expand(img0), ang0)))
    changed0 = _crop_expand(_crop(_rotate(_expand(img0), ang0)))
    changed1 = _crop_expand(_crop(_rotate(_expand(img1), ang1)))
    ang0 = image_similarity(changed0, 2)
    ang1 = image_similarity(changed1, 2)

    plotshow2(changed0, changed1, ang0, ang1)

    return ang0, ang1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Femur Bone Segmentation')

    parser.add_argument(
        '-s',
        help='Choose the image source',
        default='1_0000_p.png',
        type=str
    )
    args = parser.parse_args()
    main(args)