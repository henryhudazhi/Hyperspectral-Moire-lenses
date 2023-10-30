import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
import hdf5storage
import logging



def psnr(img1, img2):
   mse = np.mean((img1 - img2) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_3d(gt,res):
    psnr_list = []
    for k in range(gt.shape[2]):
        psnr_val = psnr(gt[:,:,k], res[:,:,k])
        psnr_list.append(psnr_val)
    psnr_mean = np.mean(np.asarray(psnr_list))
    return psnr_mean


def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:

            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# load HSIs


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def prepare_data_cave(path, file_num):
    HR_HSI = np.zeros((((512,512,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading CAVE {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_data_KAIST(path, file_num):
    HR_HSI = np.zeros((((2704,3376,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading KAIST {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_data_icvl(path, file_num):
    HR_HSI = np.zeros((((1392,1300,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading ICVL {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_test_cave(path, file_num):
    HR_HSI = np.zeros((((512,512,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading test CAVE {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_test_KAIST(path, file_num):
    HR_HSI = np.zeros((((2704,3376,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading test KAIST {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_test_icvl(path, file_num):
    HR_HSI = np.zeros((((1000,1000,25,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading test ICVL {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = hdf5storage.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI'][0:1000,0:1000,6:31]
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def checkpoint(epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    logger.info("Checkpoint saved to {}".format(model_out_path))
