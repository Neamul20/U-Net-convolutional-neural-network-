#!usr/bin/env/python3
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
'''
author : turab91
'''

class MRIDataset(Dataset):

    def __init__(self, x_data, y_data, transform=None):
        '''
        @params x_data: [n_sample, width, height]
        @params x_data: [n_sampel, width, height]
        '''

        assert x_data.ndim == 3, f'x_data expected dim 3, got {x_data.ndim}'
        assert y_data.ndim == 3, f'y_data expected dim 3, got {y_data.ndim}'

        self.x = x_data
        self.y = y_data
        self.n_samples = x_data.shape[0]
        self.transform = transform

    def __getitem__(self, index:int):

        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):

        return self.n_samples

class ToTensor:

    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)


class MulTransform:

    def __init__(self, factor):

        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        x *= self.factor
        return x, y

if __name__ == '__main__':


    x_data = np.random.randn(20, 8, 8)
    y_data = np.random.randn(20, 8, 8)
    print(f"x_data: \n{x_data[0]}")
    # ==============Dataset=================
    #mri_dataset = MRIDataset(x_data, y_data, transform=ToTensor())

    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(100)])
    mri_dataset = MRIDataset(x_data, y_data, transform=composed)


    # ===========Data Iterator===========
    train_iter = DataLoader(dataset=mri_dataset, batch_size=4, shuffle=False)


    for idx, (img_batch, mask_batch) in enumerate(train_iter):

        print(f"img: \n{img_batch[0]}")
        break

        #print(f"{idx}--img_batch: {img_batch.shape}--mask_batch: {mask_batch.shape}")

        #mask_pred = model(img_batch, mask_batch)
        #loss = loss_fn(mask_pred, mask_batch)
        #loss.backward()
        #optimizer.sero



