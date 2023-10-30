import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F

class dataset(tud.Dataset):
    def __init__(self, opt, CAVE, KAIST):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        # self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.CAVE = CAVE
        self.KAIST = KAIST
        ## load mask
        data = sio.loadmat(opt.mask_path)
        self.mask_3d = data['psf'].transpose(4,3,2,0,1)
        self.mask_3d = np.float32(self.mask_3d)

    def __getitem__(self, index):
        if self.isTrain == True:
            # index1 = 0
            index1 = random.randint(0, 26)
            d = random.randint(0, 1)
            if d == 0:
                hsi  =  self.CAVE[:,:,:,index1]
            else:
                hsi = self.KAIST[:, :, :, index1]

        else:
            index1 = index
            hsi = self.HSI[:, :, :, index1]
        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :]


        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label  =  np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()
        # n measurements
        temp = []
        for n in range(self.mask_3d.shape[0]):
            mask = np.squeeze(self.mask_3d[n,:,:,:,:])
            mask = torch.from_numpy(mask).cuda()
            Label = np.expand_dims(label, axis=0)
            Label = Label.transpose(0,3,1,2)
            Label = np.float32(Label)
            Label = torch.from_numpy(Label).cuda()
            Temp = F.conv2d(Label, mask, stride=1, padding=mask.shape[3]//2)
            temp.append(Temp)  #(b,c,h,w)

        input = torch.cat(temp, dim=1)
        # print(input.shape)
        input = input.cpu().numpy()
        # input = input.transpose(0,2,3,1)

        input = np.squeeze(input)
        input = input/np.amax(input)


        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy())
        return input, label

    def __len__(self):
        return self.num


class val_dataset(tud.Dataset):
    def __init__(self, opt, CAVE):
        super(val_dataset, self).__init__()
        self.size = opt.size
        self.num = opt.testset_num
        self.CAVE = CAVE

        ## load mask
        data = sio.loadmat(opt.mask_path)
        self.mask_3d = data['psf'].transpose(4,3,2,0,1)
        self.mask_3d = np.float32(self.mask_3d)

    def __getitem__(self, index):
        hsi = self.CAVE[:, :, :, index]
        shape = np.shape(hsi)
        label = hsi


        # n measurements
        temp = []
        for n in range(self.mask_3d.shape[0]):
            mask = np.squeeze(self.mask_3d[n,:,:,:,:])
            mask = torch.from_numpy(mask).cuda()
            Label = np.expand_dims(label, axis=0)
            Label = Label.transpose(0,3,1,2)
            Label = np.float32(Label)
            Label = torch.from_numpy(Label).cuda()
            Temp = F.conv2d(Label, mask, stride=1, padding=mask.shape[3]//2)
            temp.append(Temp)

        input = torch.cat(temp, dim=1)
        # print(input.shape)
        input = input.cpu().numpy()
        # input = input.transpose(0,2,3,1)

        input = np.squeeze(input)
        input = input/np.amax(input)


        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy())
        return input, label

    def __len__(self):
        return self.num
