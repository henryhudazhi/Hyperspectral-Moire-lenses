import torch
import os
import argparse
# from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import hdf5storage
import torch.nn.functional as F
from utils1 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='/workspace/huhq/MST/k_measurement/testdata1/cave/', type=str,help='path of data')
# parser.add_argument('--data_path', default='/workspace/huhq/MST/k_measurement/testdata/', type=str,help='path of data')
parser.add_argument('--mask_path', default='/workspace/huhq/MST/k_measurement/psf_4k.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=6, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
opt = parser.parse_args()
print(opt)

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((512,512,25,file_num))))
    # HR_HSI = np.zeros((((1392,1300,25,file_num))))
    # HR_HSI = np.zeros((((2704,3376,25,file_num))))
    file_list = os.listdir(path)

    for idx in range(file_num):
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        # print(path1)
        data = hdf5storage.loadmat(path1)
        # HR_HSI[:,:,:,idx] = data['HSI'][:1024,:1024,6:31]
        HR_HSI[:,:,:,idx] = data['HSI'][:,:,6:31]
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_psf(path):
    ##  load mask
    data = sio.loadmat(path)
    mask_3d = data['psf'].transpose(4,3,2,0,1)
    mask_3d = np.float32(mask_3d)
    # mask_3d = torch.from_numpy(mask_3d).cuda()
    return mask_3d

HR_HSI = prepare_data(opt.data_path, 3)
mask_3d = load_psf(opt.mask_path)
# mask_3d = load_psf(opt.mask_path)[1,:,:,:,:]
# mask_3d = np.expand_dims(mask_3d, axis=0)

# pretrained_model_path = "/workspace/huhq/MST/k_measurement/exp/mst_plus_plus1/550-700/model_300.pth"
pretrained_model_path = "/workspace/huhq/MST/k_measurement/exp/mst_s1/psf_4k/model_300.pth"
save_path = pretrained_model_path.replace(pretrained_model_path.split('/')[-1], 'result/')
model = torch.load(pretrained_model_path)
model = model.eval()
# model = dataparallel(model, 1)
psnr_total = 0
k = 0
for j in range(3):
    with torch.no_grad():
        meas = HR_HSI[:,:,:,j]
        meas = torch.FloatTensor(meas)
        meas = meas.unsqueeze(0).permute(0,3,1,2)
        print(meas.shape)
        meas = meas.cuda()
        temp = []
        for n in range(mask_3d.shape[0]):
            mask = np.squeeze(mask_3d[n,:,:,:,:])
            mask = torch.from_numpy(mask).cuda()
            Temp = F.conv2d(meas, mask, stride=1, padding=mask.shape[3]//2)
            temp.append(Temp)

        input = torch.cat(temp, dim=1)
        input = input.cpu().numpy()
        # input = np.squeeze(input)
        input = input/np.amax(input)
        input = torch.FloatTensor(input.copy())
        input = Variable(input)
        input = input.cuda()
        print(input.shape)
        # mask_3d_shift = mask_3d_shift.cuda()
        # mask_3d_shift_s = mask_3d_shift_s.cuda()
        out = model(input)
        result = out
        result = result.clamp(min=0., max=1.)
        # result[result < 0.] = 0.
        # result[result > 1.] = 1.
    k = k + 1
    if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
        os.makedirs(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    gt = meas.cpu().permute(2,3,1,0).squeeze(3).numpy()
    PSNR = psnr_3d(gt, res)

    print('test PSNR: ',PSNR)
    save_file = save_path + f'{j}.mat'
    sio.savemat(save_file, {'res':res, 'gt':gt})
