import os
import time
import argparse
import numpy as np


from tqdm import tqdm
from PIL import Image
from cv2 import imwrite

from functools import reduce
from shutil import copyfile
from multiprocessing import Pool

from options.option import Option_Dict
from options import option

from core import trainer
from core.utils import preprocess
from core.models.model_factory import Model

from utils.util import calc_pad, pad_forward, pad_inv, init_preprocess_opt, get_init_method, save_color_img, hist_match_reverse, de_nor, calc_pad
from utils.evaluation import Evaluator, EmptyEvaluator
from calc_test import calc_ssim_psnr_test, scan_dir, create_gif_test
from utils.metrics.ssim import SSIM
from data.radar_dataset import create_dataloader, RadarData, RadarSatData

import torch
# from torch.utils.tensorboard import SummaryWriter
# -----------------------------------------------------------------------------
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--is_real_testing', type=int, default=0)
args = parser.parse_args()
opt = Option_Dict(option.parse(args.opt, is_training=args.is_training, is_real_testing=args.is_real_testing))
batch_size = opt.batch_size
opt.dist = False
init_preprocess_opt(opt)
print("opt:", opt)
print("Padding:", opt.padding)
print("img_width:", opt.img_width)
print("img_height:", opt.img_height)

# Create Dataset
if opt.dataset_name == 'radar':
    train_data = RadarData(
        data_root=opt.train_data_root,
        mode='train',
        in_seq=opt.input_length,
        out_seq=opt.total_length - opt.input_length,
        sat=opt.sat,
        dem=opt.dem,
        month=opt.month,
    )
    test_data = RadarData(
        data_root=opt.test_data_root,
        mode='test',
        in_seq=opt.input_length,
        out_seq=opt.total_length - opt.input_length, 
        sat=opt.sat,
        dem=opt.dem,
        month=opt.month,
    )

elif opt.dataset_name == 'radar_sat':
    train_data = RadarSatData(
        data_root=opt.train_data_root,
        mode='train',
        radar_in_seq=opt.input_length,
        sat_in_seq=opt.sat_input_length,
        out_seq=opt.total_length - opt.input_length, 
        dem=opt.dem,
        sat = opt.sat,
        time = opt.time,
    )

    test_data = RadarSatData(
        data_root=opt.test_data_root,
        mode='test',
        radar_in_seq=opt.input_length,
        sat_in_seq=opt.sat_input_length,
        out_seq=opt.total_length - opt.input_length,
        dem=opt.dem,
        sat=opt.sat, 
        time = opt.time,
    )
else:
    raise Exception("wrong dataset!")


train_loader = create_dataloader(train_data, mode='train', opt=opt)
test_loader = create_dataloader(test_data, mode='test', opt=opt)
if opt.model_name[:3] == 's2s':
    schedule_sampling_or_init_h = get_init_method(opt, 's2s')
else:
    schedule_sampling_or_init_h = get_init_method(opt, 'schedule_sampling')

def wrapper_test(model, it_num, color=True, gray=False, save_root='', is_real = False, name = None):
    if name is not None:
        save_root = os.path.join(save_root, name)
    else :
        save_root = os.path.join(save_root, it_num)
    model.to_eval()
    # 用于染色
    task_pool = Pool(8)

    mse = 0
    mae = 0
    psnr = 0
    ssim = 0
    count = 0
    batch_size = opt.test_batch_size

    _, real_input_flag = schedule_sampling_or_init_h(0, 1e9, batch_size=batch_size)
    output_length = opt.total_length - opt.input_length

    if save_root is not None:
        if opt.pretrained_model is not None:
            path_name = opt.pretrained_model[opt.pretrained_model.rindex('/') + 1:opt.pretrained_model.rindex('.')].split('-')[1]
        else:
            path_name = it_num
        evaluator = Evaluator(os.path.join(opt.metric_dir, path_name), seq=output_length)
    else:
        evaluator = EmptyEvaluator()
    test_loader_ = None
    test_len = 0
    pad = None
    is_pad = False
 
    test_loader_ = test_loader
    test_len = len(test_loader)   
    pad = opt.padding    
    is_pad = opt.pad 
    with torch.no_grad():
        for dat_indexes in tqdm(test_loader_, total = test_len):
      
            indexes = dat_indexes[-1]
            ims = dat_indexes[:-1]
            if not isinstance(ims, list):
                ims = [ims]

            for i in range(len(ims)):
                if is_pad:
                    ims[i] = pad_forward(ims[i], pad)             
                if opt.patch_size > 1:
                    ims[i] = preprocess.reshape_patch(ims[i], opt.patch_size)              
            img_gen, losses = model.test(ims, real_input_flag)
            if opt.patch_size > 1:
                img_gen = preprocess.reshape_patch_back(img_gen, opt.patch_size)
            img_out = img_gen[:, -output_length:]
            if is_pad:
                img_out = pad_inv(img_out, pad)
                ims[0] = pad_inv(ims[0], pad)
            last_inputs = de_nor(ims[0][:,-output_length - 1].detach().cpu().numpy())
            tars = de_nor(ims[0][:, -output_length:].detach().cpu().numpy())
      
            if opt.patch_size > 1:
                tars = preprocess.reshape_patch_back(tars, opt.patch_size)
            tars = tars[..., 0]
            img_out = img_out[..., 0]
            last_inputs = last_inputs[..., 0]
            evaluator.evaluate(tars, img_out)
            # mae += np.mean(np.abs(tars - img_out))
            # mse += np.mean(np.square(tars - img_out))
            mae += losses['mae']
            mse += losses['mse']
            psnr += losses['psnr']
            ssim += losses['ssim']
            count = count + 1

            indexes = indexes.numpy()
            if save_root is not None:
                # 图像预处理
                # img_out[img_out > 64] = 64
                # img_out[img_out < 15] = 0 
                img_out  = np.ceil(img_out) 
                img_out = img_out.astype(np.uint8) 
                
                tars = tars.astype(np.uint8)
                
                last_inputs = last_inputs.astype(np.uint8)
                for batch_idx in range(batch_size):
                    # 如果高回波太少，则不进行保存
                    # if (last_inputs[batch_idx] > 45).sum() <= 3000:
                    #     continue
                    sample_index_name = indexes[batch_idx]
                    pred_img_fold = os.path.join(save_root, str(sample_index_name), "pred")
                    gt_img_fold = os.path.join(save_root, str(sample_index_name), "gt")
                    if gray:
                        os.makedirs(pred_img_fold, exist_ok=True)
                        os.makedirs(gt_img_fold, exist_ok=True)

                    color_gt_img_fold = os.path.join(save_root, str(sample_index_name), "color_gt")
                    color_pred_img_fold = os.path.join(save_root, str(sample_index_name), "color_pred")
                    hist_img_fold = os.path.join(save_root, str(sample_index_name), "hist")
                    if color:
                        os.makedirs(color_gt_img_fold, exist_ok=True)
                        os.makedirs(color_pred_img_fold, exist_ok=True)
                        os.makedirs(hist_img_fold, exist_ok=True)
                    
                    bat_img_out = img_out[batch_idx]
                    bat_tar = tars[batch_idx]
                    last_input = last_inputs[batch_idx]
                    
                    target_hist = np.array(Image.fromarray(last_input).histogram())

                    for ot in range(output_length):
                        t = ot + opt.input_length + 1

                        # 保存灰度图
                        if gray:
                            import pdb
                            # pdb.set_trace()
                            img_path = os.path.join(pred_img_fold, f"{t}.png")
                            imwrite(img_path, bat_img_out[ot])
                            img_path = os.path.join(gt_img_fold, f"{t}.png")
                            imwrite(img_path, bat_tar[ot])

                        if color:
                            # 未规定化的染色
                           
                            img_path = os.path.join(color_pred_img_fold, f"{t}.png")
                            task_pool.apply_async(save_color_img, args=(img_path, bat_img_out[ot]))
                            
                            # # 规定化的染色
                            # hist_image = hist_match_reverse(bat_img_out[ot], target_hist)
                         
                            # img_path = os.path.join(hist_img_fold, f"{t}.png")
                            # task_pool.apply_async(save_color_img, args=(img_path, hist_image))
                            
                            # 实况的染色
                            img_path = os.path.join(color_gt_img_fold, f"{t}.png")
                            task_pool.apply_async(save_color_img, args=(img_path, bat_tar[ot]))
    mae /= count
    mse /= count
    psnr /= count
    ssim /= count
    print(f"mae: {mae} | mse: {mse} | psnr: {psnr} | ssim: {ssim}")

    task_pool.close()
    task_pool.join()
    evaluator.record(mae, mse, psnr, ssim)
    csi = evaluator.done()
    print(csi)
    return {
        "test_mae": mae,
        "test_mse": mse,
        "test_psnr": psnr,
        "test_ssim": ssim,
        "test_csi": csi
    }

def wrapper_valid(model):
    pass

def wrapper_train(model):
    model.to_train()
    # writer = SummaryWriter(opt.log_dir)
    eta = opt.sampling_start_value
    itr = opt.start_itr
    for _ in range(opt.epochs):
        loss = []
        for _, ims in tqdm(enumerate(train_loader), total = len(train_loader)):
            itr += 1
            if not isinstance(ims, list):
                ims = [ims]

            for i in range(len(ims)):
                if opt.pad:
                    ims[i] = pad_forward(ims[i], opt.padding)
                # 雷达图片不需要切割，本身图片就是384*384大小的
                if opt.patch_size > 1:
                    ims[i] = preprocess.reshape_patch(ims[i], opt.patch_size)
            # 初始化hidden state
            eta, real_input_flag = schedule_sampling_or_init_h(eta, itr)
            # begin_time = time.time()
            cost = trainer.train(model, ims, real_input_flag, opt, itr)
            # print("train time", time.time() - begin_time)
            loss.append(cost)
            if itr % opt.display_interval == 0:
                print(f'itr: {itr}')
                loss_len = len(loss)
                loss = reduce(lambda x, y: {k: x[k]+y[k] for k in x}, loss)
                loss = {k: v / loss_len for k, v in loss.items()}
                print(f'train_loss: {loss}')
                for k, v in loss.items():
                    # writer.add_scalar(k, v, itr)
                    print(k, v, itr)
                loss = []
      
            if itr % opt.test_interval == 0 and itr >= opt.snapshot_interval:
                losses = wrapper_test(model, str(itr), color = False, gray = False, save_root=opt.result_dir, is_real = False)
                model.save(itr, losses['test_csi'])
                print('====== the test loss is ======', str(losses))
                for k, v in losses.items():
                    if k != "test_csi":
                        # writer.add_scalar(k, v, itr)
                        print(k, v, itr) 

            if itr > opt.early_stop:
                break
            
def check_dirs(opt):
    opt.log_dir = os.path.join(opt.result_root, "logs", opt.exp_name)
    opt.checkpoint_dir = os.path.join(opt.result_root, "checkpoints", opt.exp_name)
    opt.result_dir = os.path.join(opt.result_root, "results", opt.exp_name)
    opt.config_dir = os.path.join(opt.result_root, "configs", opt.exp_name)
    opt.metric_dir = os.path.join(opt.result_root, "metrics", opt.exp_name)
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.config_dir, exist_ok=True)
    os.makedirs(opt.metric_dir, exist_ok=True)
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
check_dirs(opt)
setup_seed(42)
print('Initializing models')
model = Model(opt)

print("opt.is_training:", opt.is_training)
print("opt.is_real_testing:", opt.is_real_testing)
if opt.pretrained_model:
    model.load(path=opt.pretrained_model, strict=opt.strict)

if opt.is_training:
    # 拷贝配置文件
    opt_name = args.opt[args.opt.rindex('/') + 1:]
    copyfile(args.opt, os.path.join(opt.config_dir, opt_name))
    wrapper_train(model)
else:
    if opt.pretrained_model is not None:
        it_num = opt.pretrained_model[opt.pretrained_model.rindex('/') + 1:opt.pretrained_model.rindex('.')].split('-')[1]
        wrapper_test(model, it_num, save_root=opt.result_dir, gray = True, color = False, is_real = opt.is_real_testing, name = "multi_attn_concat")        
    # calc_ssim_psnr_test(opt.save_root)
