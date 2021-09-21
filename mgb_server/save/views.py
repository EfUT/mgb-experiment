from math import asin, cos, pi, sin, sqrt, tan
from django.shortcuts import render
import cv2
import numpy as np

from .forms import ImageForm
from cv.process import to_fish, AOV
from save.models import CapturedImage
from solve.nonlinear_solver import solve_nonlin


def upload_file(request):
    if request.method == 'POST':
        print(request.POST)
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            
            # 1. 観測値による理論値
            ox, oy, oz = _get_observed_position(float(form.cleaned_data['a']), float(form.cleaned_data['b']), float(form.cleaned_data['c']), float(form.cleaned_data['h']), float(form.cleaned_data['imaginary_planes_angle']))
            
            
            # 2. 画像による推測値
            img_obj = form.instance
            img = cv2.imread(img_obj.image.path)
            
            # 正方にトリミング
            if img.shape[0] > img.shape[1]:
                img = img[img.shape[0] // 2 - img.shape[1] // 2 : img.shape[0] // 2 + img.shape[1] // 2, :]
            elif img.shape[0] < img.shape[1]:
                img = img[:, img.shape[1] // 2 - img.shape[0] // 2 : img.shape[1] // 2 + img.shape[0] // 2]
            
            fish_img = to_fish(img)
            print('returned')
            fish_img = cv2.imread('result.jpg')
            
            # 赤の抽出
            red_mask, _ = _detect_color(fish_img, color='red')
            # cv2.imwrite('red_mask.png', red_mask)
            img_with_contours, contour_areas, center_red, err_red = _get_contour(red_mask)
            # cv2.imwrite('result.png', img_with_contours)
            
            # 緑の抽出
            blue_mask, _ = _detect_color(fish_img, color='green')
            img_with_contours, contour_areas, center_blue, err_blue = _get_contour(blue_mask)
            
            print('center:', center_blue, center_red)
            
            # 位置算出のための各種変数の用意
            center = np.array((fish_img.shape[0] // 2, fish_img.shape[1] // 2))
            ra = np.linalg.norm(center_red - center)
            rb = np.linalg.norm(center_blue - center)
            AB = np.linalg.norm(center_red - center_blue)
            
            print(center, ra, rb, AB)
            
            R = fish_img.shape[0] / 2 / tan(AOV / 2)
            
            theta = pi * ra / 2 / R
            phi = pi * rb / 2 / R
            
            psi = (rb**2 + ra**2 - AB**2) / (2 * ra * rb)
            
            print(R, theta, phi, psi)
            
            accx = float(form.cleaned_data['x'])
            accy = float(form.cleaned_data['y'])
            accz = float(form.cleaned_data['z'])
            
            alpha = asin(accy / sqrt(accy**2 + accz**2)) + float(form.cleaned_data['imaginary_planes_angle'])
            beta = asin(accx / sqrt(accx**2 + accz**2))
            
            a = float(form.cleaned_data['a'])
            
            print(accx, accy, accz, alpha, beta, a)
            
            # 非線形方程式を解く
            
            pos_result = solve_nonlin({
                'angle': [theta, phi, psi],
                'grad': [alpha, beta],
                'a': a
            })
            
            
            ix, iy, iz = pos_result[0], pos_result[1], pos_result[2]
            
            print('root:', ix, iy, iz)
            
            
            return render(request, 'index.html', {
                'form': form,
                'title': form.cleaned_data['title'],
                'raw_img': img_obj.image,
                'center_red': center_red,
                'center_blue': center_blue,
                'err_red': err_red,
                'err_blue': err_blue,
                'ox': round(ox, 2),
                'oy': round(oy, 2),
                'oz': round(oz, 2),
                'ix': round(ix, 2),
                'iy': round(iy, 2),
                'iz': round(iz, 2),
            })
        else:
            print(form.errors)
    else:
        form = ImageForm()
    return render(request, 'index.html', context={'form': form})


def _get_observed_position(a, b, c, h, im_angle):
    im_angle = im_angle / 180 * pi
    lx = (b**2 - c**2) / (4 * a)
    ly = sqrt((4 * a**2 + b**2 - c**2 + 4 * a * b) * (-4 * a**2 - b**2 + c**2 + 4 * a * b)) / (4 * a)
    return lx, ly * cos(im_angle) -  h * sin(im_angle), ly * sin(im_angle) + h * cos(im_angle)
    


def _detect_color(image: np.ndarray, color='red'):
    '''
    色を抽出する
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if color == 'red':
        hsv_min = np.array([0, 64, 100])
        hsv_max = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, hsv_min, hsv_max)
        
        hsv_min = np.array([170, 64, 100])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
        
        mask = mask1 + mask2
    
    elif color == 'green':
        hsv_min = np.array([30, 64, 100])
        hsv_max = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    else:
        mask = None
    
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    return mask, masked_img


def _get_contour(image):
    '''
    輪郭の抽出
    '''
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return image, None, None, 'Error: Cannot find any contours.'
    contour = [c[0] for c in contours[0]]
    center = np.round(np.mean(contour, axis=0))
    cv2.drawContours(image, contours, -1, color=(0, 0, 255), thickness=2)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    return image, contour_areas, center, ''