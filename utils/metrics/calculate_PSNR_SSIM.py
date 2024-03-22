'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import json

def calc_ssim_psnr(folder_GT, folder_Gen, crop_border=0, gray=True, start_calc_num=5, json_path=None):
    # Configurations
    suffix = ''  # suffix for Gen images
    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/**/*.png', recursive=True))
    sample_dirs = os.listdir(folder_Gen)
    model_name = folder_Gen.split('/')[1]
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            sample_metric = json.load(json_file)
            sample_metric[model_name] = {}
    else:    
        sample_metric = {}
        sample_metric[model_name] = {}
    for sample_name in sample_dirs:
        sample_metric[model_name][sample_name] = {'ssim': {}, 'psnr': {}, 'ssim_mean':0., 'psnr_mean':0.}
    for i, img_path in enumerate(img_list):
        sample_name = os.path.dirname(img_path).split('/')[-1]        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if int(base_name.split('_')[-1]) < start_calc_num:
            continue
        folder_Sample_Gen = os.path.join(folder_Gen, sample_name)
        if gray:
            im_GT = cv2.imread(img_path, 0)
            print("img_gen path:", os.path.join(
                folder_Sample_Gen, base_name + suffix + '.png'))
            im_Gen = cv2.imread(os.path.join(
                folder_Sample_Gen, base_name + suffix + '.png'), 0)
        else:
            im_GT = cv2.imread(img_path)
            im_Gen = cv2.imread(os.path.join(
                folder_Sample_Gen, base_name + suffix + '.png'))

        im_GT_in = im_GT
        im_Gen_in = im_Gen

        if crop_border > 0:
            cropped_GT = im_GT_in[crop_border:-
                                  crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border: -
                                    crop_border, crop_border: - crop_border]
        else:
            cropped_GT = im_GT_in
            cropped_Gen = im_Gen_in
        if im_GT_in.ndim != 2 and im_GT_in.ndim != 3:
            raise ValueError(
                'Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT, cropped_Gen)
        SSIM = calculate_ssim(cropped_GT, cropped_Gen)
        sample_metric[model_name][sample_name]['ssim'][base_name] = SSIM
        sample_metric[model_name][sample_name]['psnr'][base_name] = PSNR
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    for k, v in sample_metric[model_name].items():
        sample_metric[model_name][k]['ssim_mean'] = np.mean(list(sample_metric[model_name][k]['ssim'].values()))
        sample_metric[model_name][k]['psnr_mean'] = np.mean(list(sample_metric[model_name][k]['psnr'].values()))
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(sample_metric, f)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    # calc_ssim_psnr('/mnt/A/satelite/datasets/test',
    #                '/home/ices/yl/SatelliteSP/Invconvgru_test')
    calc_ssim_psnr('/mnt/A/satelite/datasets/test',
                   '/home/ices/yl/SatelliteSP/results/Invconvgru_test_inv3')
