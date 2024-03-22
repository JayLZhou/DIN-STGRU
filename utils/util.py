import torch
import os
import shutil

import numpy as np
import torch.nn.functional as F
import copy

from PIL import Image
from utils.color_map import color_map
# from utils.color_map_paper import color_map
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def cmp_fuc(a, b):
    if len(a) == len(b):
        if a == b:
            return 0
        elif a < b:
            return -1
        else:
            return 1
    else:
        if len(a) < len(b):
            return -1
        else:
            return 1

MAX_VAL = 74

def nor(x):
    h = MAX_VAL / 2
    x = (x.astype(np.float32) - h) / h
    return x
def readimgs(files):
    images = []
    for f in files:
        data = np.array(Image.open(f))
        data[data < 15]  = 0
        data[data > 74] = 0
        images.append(data)
    return np.array(images)

def de_nor(x):
    h = MAX_VAL / 2
    x = (x * h) + h
    return x
def de_nor_ckim(x):
    h = MAX_VAL / 2
    x = (x * h) + h
    return x


# def nor_sat(x):
#     h = 255 / 2
#     x = (x.astype(np.float32) - h) / h
#     return x

# 染色函数
def gray2color(img):
    h, w = img.shape
    new_img = np.zeros((h, w, 4), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = color_map[img[i, j]]
    img = Image.fromarray(new_img, mode="RGBA")
    return img

def save_color_img(img_path, img):
    try:
        gray2color(img).save(img_path)
    except Exception as e:
        print(e)

# 直方图匹配函数，接受原始图像和目标灰度直方图 orig_img[700,900],tgt_hist[256]
def hist_match_reverse(orig_img, tgt_hist):
    
    orig_img[orig_img >= 65] = 64
    orig_img[orig_img < 15] = 0
    input_max=64
    for i in range(64, -1, -1):
        if tgt_hist[i] > 5:
            input_max = i
            break
    # print(input_max)
    orig_hist = np.array(Image.fromarray(orig_img).histogram())
    orig_hist[0] = 0
    # 分别对orig_acc和tgt_acc归一化
    orig_sum = 0.0
    tgt_sum = 0.0
    # print('input max',input_max)
    for i in range(1, input_max):
        orig_sum += orig_hist[i]
        tgt_sum += tgt_hist[i]
    # for i in range(1, 65):
    orig_hist= orig_hist/(orig_sum+1)
    tgt_hist=tgt_hist/ tgt_sum
    # 计算累计直方图
    tmp = 0.0
    tgt_acc = tgt_hist.copy()

    for i in range(input_max, 0,-1):
        tmp += tgt_hist[i]
        tgt_acc[i] = tmp
    tmp = 0.0
    orig_acc = orig_hist.copy()
    for i in range(input_max,0, -1):
        tmp += orig_hist[i]
        orig_acc[i] = tmp

    # 计算映射
    M = np.zeros(65,dtype=np.uint8)
    M[1:] = 64
    for i in range(input_max, 0,-1):
        idx = input_max
        minv = 1
        for j in range(input_max, 0,-1):
            if np.fabs(tgt_acc[j] - orig_acc[i]) < minv:
                # update the value of minv
                minv = np.fabs(tgt_acc[j] - orig_acc[i])
                idx = int(j)
        # if idx-M[i-1]>2:
        #     M[i]=M[i-1]+1
        # else:
        M[i] = idx
        # M stores the index of closest tgt_hist gray value
    # print(M)
    orig_img=orig_img.astype(np.int)
    des = M[orig_img]
    return des


def calc_pad(padding_W, padding_H):
    padding_left, padding_right = padding_W // 2, padding_W // 2
    padding_top, padding_bottom = padding_H // 2, padding_H // 2
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    return padding


# 扩充 TODO
# x: [b, c, h , w]
# padding: (left, right, top, bottom)
def pad_forward(x, padding):
    # b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    padding_left, padding_right, padding_top, padding_bottom = padding
    if isinstance(x, torch.Tensor):
        pad_x = -1 * torch.ones((x.shape[0], x.shape[1], x.shape[2] + padding_top + padding_bottom,
                             x.shape[3] + padding_left + padding_right, x.shape[4]))
    elif isinstance(x, np.ndarray):
        pad_x = -1 * torch.ones((x.shape[0], x.shape[1], x.shape[2] + padding_top + padding_bottom,
                          x.shape[3] + padding_left + padding_right, x.shape[4]))
    else:
        assert False, "Invalid Input Type: pad obj must be a numpy array or torch Tensor"

    if padding_bottom == 0 and padding_right == 0:
        pad_x[:, :, padding_top:, padding_left:] = x
    elif padding_right == 0:
        pad_x[:, :, padding_top:-padding_bottom, padding_left:] = x
    elif padding_bottom == 0:
        pad_x[:, :, padding_top:, padding_left:-padding_right] = x
    else:
        pad_x[:, :, padding_top:-padding_bottom,
              padding_left: - padding_right] = x
    return pad_x


def pad_inv(x, padding):
    padding_left, padding_right, padding_top, padding_bottom = padding
    if padding_bottom == 0 and padding_right == 0:
        return x[:, :, padding_top:, padding_left:]
    elif padding_right == 0:
        return x[:, :, padding_top:-padding_bottom, padding_left:]
    elif padding_bottom == 0:
        return x[:, :, padding_top:, padding_left:-padding_right]
    else:
        return x[:, :, padding_top:-padding_bottom,
                 padding_left:-padding_right]

# 染色函数
def mapping(img):
    h, w = img.shape
    new_img = np.zeros((h, w, 4), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = color_map[img[i, j]]
    img = Image.fromarray(new_img, mode="RGBA")
    return img
def normalization(frames, up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up / 2)
    new_frames -= 1
    return new_frames


def denormalization(frames, up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up / 2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames


def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def get_init_method(opt, name='schedule_sampling'):
    def schedule_sampling(eta, itr, batch_size=opt.batch_size):
        img_height = opt.img_height
        img_width = opt.img_width
        patch_size = opt.patch_size
        channel = opt.img_channel

        zeros = np.zeros(
            (batch_size, opt.total_length - opt.input_length - 1, 1, 1, 1))
        if itr >= opt.sampling_stop_iter or not opt.scheduled_sampling:
            return 0.0, zeros

        if itr < opt.sampling_stop_iter:
            eta -= opt.sampling_changing_rate
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (batch_size, opt.total_length - opt.input_length - 1))
        true_token = (random_flip < eta)
        ones = 1.
        zeros = 0.
        real_input_flag = []
        for i in range(batch_size):
            for j in range(opt.total_length - opt.input_length - 1):
                if true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
        real_input_flag = np.array(real_input_flag)
        real_input_flag = np.reshape(
            real_input_flag,
            (batch_size, opt.total_length - opt.input_length - 1, 1, 1, 1))
        return eta, real_input_flag

    def s2s_init_h(eta, itr, batch_size=opt.batch_size):
        channel = opt.decode_in_channel
        h, w = opt.decode_in_shape
        zeros = np.zeros((batch_size, opt.total_length - opt.input_length, h , w, channel))
        return 0.0, zeros

    if name == 's2s':
        return s2s_init_h
    elif name == 'schedule_sampling':
        return schedule_sampling
    else:
        raise NotImplementedError(
            "only choose s2s_init_h or schedule_sampling")


def init_preprocess_opt(opt):
    opt.early_stop = opt.max_iterations
    if opt.is_real_testing == 0:
        if opt.padding_W is None:
            opt.padding_W = 0
        if opt.padding_H is None:
            opt.padding_H = 0
        if opt.padding_W == 0 and opt.padding_H == 0:
            opt.pad = False
        else:
            opt.pad = True
            opt.padding = calc_pad(padding_W=opt.padding_W,
                                padding_H=opt.padding_H)
            opt.img_width = opt.img_width + opt.padding_W
            opt.img_height = opt.img_height + opt.padding_H
    else :
        if opt.test_pad_W is None:
            opt.test_pad_W = 0
        if opt.test_pad_H is None:
            opt.test_pad_H = 0
        if opt.test_pad_W == 0 and opt.test_pad_H == 0:
            opt.pad = False
        else:
            opt.pad = True
            opt.padding = calc_pad(padding_W = opt.test_pad_W,
                                padding_H = opt.test_pad_H)
            opt.img_width = opt.test_width + opt.test_pad_W
            opt.img_height = opt.test_height + opt.test_pad_H
if __name__ == '__main__':
    pass
