from math import atan, sqrt, pi, tan
import numpy as np
import cv2
from numpy.lib.index_tricks import nd_grid

np.set_printoptions(threshold=np.inf)

# img = cv2.imread('test.jpeg')
# height, width, ch = img.shape

# 画角（radian）
AOV = 80 * pi / 180
# R = pi * width / 2 / AOV


def convert_position(x, y, shape):
    '''
    魚眼のpixel毎の位置変換
    '''
    x = x - shape[0] // 2
    y = -(y - shape[0] // 2)
    
    r = sqrt(x**2 + y**2)
    # R = pi * shape[0] / 2 / AOV
    R = shape[0] / 2 / tan(AOV / 2)
    
    if r == 0:
        _x = _y = 0
    else:
        _x = 2 * R / pi * x / r * atan(r/R)
        _y = 2 * R / pi * y / r * atan(r/R)
    
    return round(_x) + shape[0] // 2, -round(_y) + shape[0] // 2

pos_x = None
pos_y = None


def to_fish(img: np.ndarray):
    '''
    魚眼への変換
    '''
    global pos_x, pos_y
    new_img = np.full(img.shape, 0, np.float32)

    
    if pos_x is None:
        pos_x, pos_y = np.zeros(img.shape[0]), np.zeros(img.shape[1])
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pos = convert_position(x, y, img.shape)
                pos_x[x], pos_y[y] = pos[0], pos[1]
                new_img[pos[0], pos[1], :] = img[x, y, :]
            
            if x % 10 == 0:
                print(f'x: {x}')
        print('convert finished')
        # with open('map_old.txt', 'w') as f:
        #     f.write(np.array_str(pos_x))
        #     f.write(np.array_str(pos_y))
    
    else:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                new_img[pos_x, pos_y, :] = img[x, y, :]
            
            if x % 10 == 0:
                print(f'x: {x}')
        
    cv2.imwrite('result.jpg', new_img)
    return new_img


# img = cv2.imread('test.jpeg')
# img = to_fish(img)

# cv2.imwrite('result.jpeg', img)