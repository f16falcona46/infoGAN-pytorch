import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

import numpy as np
from model import *


def save_single_image(img, fname):
    from PIL import Image
    ndarr = img.mul(255).clamp(0, 255).byte()[0].permute(0, 1).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(fname)

class Trainer:
    def __init__(self, G, FE, D, Q, Discrete_Vars, Continuous_Vars, Noise_Vars, Image_Width, Image_Height, batch_size, OCT = False):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q

        self.Discrete_Vars = Discrete_Vars
        self.Continuous_Vars = Continuous_Vars
        self.Total_Vars = Discrete_Vars + Continuous_Vars + Noise_Vars

        self.Continuous_Steps = 10

        self.Noise_Vars = Noise_Vars

        self.batch_size = batch_size
        self.Progress_Batch_Size = 10
        self.Image_Width = Image_Width
        self.Image_Height = Image_Height

        self.OCT = OCT

        self.dis_c_tensor = torch.FloatTensor(self.Progress_Batch_Size, self.Discrete_Vars).cuda()
        self.con_c_tensor = torch.FloatTensor(self.Progress_Batch_Size, self.Continuous_Vars).cuda()
        self.noise_tensor = torch.FloatTensor(self.Progress_Batch_Size, self.Noise_Vars).cuda()

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(self.Discrete_Vars, size=bs)
        c = np.zeros((bs, self.Discrete_Vars))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.Total_Vars, 1, 1)

        return z, idx

    def compress_and_save(self, x, idx):
        fix_noise = torch.Tensor(self.Progress_Batch_Size, self.Noise_Vars).zero_()
        self.dis_c_progress.data.resize_(self.Progress_Batch_Size, self.Discrete_Vars)
        self.con_c_progress.data.resize_(self.Progress_Batch_Size, self.Continuous_Vars)
        self.noise_progress.data.resize_(self.Progress_Batch_Size, self.Noise_Vars)
        
        #sel_x = x[idx]
        #x = torch.stack([sel_x for _ in range(self.Progress_Batch_Size)])
        fe_out = self.FE(x.cuda())
        c_d, c_mu, c_var = self.Q(fe_out)
        one_hot = np.zeros((self.Progress_Batch_Size, self.Discrete_Vars))
        for i in range(self.Progress_Batch_Size):
            one_hot[i, c_d.argmax(1)[i]] = 1.0
        self.dis_c_progress.data.copy_(torch.tensor(one_hot).cuda())
        self.con_c_progress.copy_(c_mu)
        self.noise_progress.data.copy_(fix_noise)
        z = torch.cat([self.noise_progress, self.dis_c_progress, self.con_c_progress], 1).view(-1, self.Total_Vars, 1, 1)
        f_x = self.G(z)

        self.con_c_progress.zero_()
        z = torch.cat([self.noise_progress, self.dis_c_progress, self.con_c_progress], 1).view(-1, self.Total_Vars, 1, 1)
        f_x_z = self.G(z)

        save_image(x, "./tmp/real_row.png", self.Progress_Batch_Size)
        save_image(f_x, "./tmp/fake_row_actual_cont.png", self.Progress_Batch_Size)
        save_image(f_x_z, "./tmp/fake_row_cont_zeroed.png", self.Progress_Batch_Size)

    def run(self):
        if self.OCT:
            OCT_images = np.load("./A/np/train/all.npy") / (1<<16)
            dataset = TensorDataset(torch.Tensor(OCT_images))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        else:
            dataset = dset.MNIST(
                './dataset', transform=transforms.ToTensor(), download=True)
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
        self.dis_c_progress = Variable(self.dis_c_tensor)
        self.con_c_progress = Variable(self.con_c_tensor)
        self.noise_progress = Variable(self.noise_tensor)

        for num_iters, batch_data in enumerate(dataloader, 0):
            if self.OCT:
                x = batch_data[0]
            else:
                x, _ = batch_data
            self.compress_and_save(x, 0)
            break

def main():
    Discrete_Vars = 10
    Continuous_Vars = 4
    Noise_Vars = 64
    OCT = False

    fe = FrontEnd(OCT)
    fe.load_state_dict(torch.load("netFE_epoch_036_iter_04.pth"))
    d = D()
    d.load_state_dict(torch.load("netD_epoch_036_iter_04.pth"))
    q = Q(Discrete_Vars, Continuous_Vars)
    q.load_state_dict(torch.load("netQ_epoch_036_iter_04.pth"))
    g = G(Discrete_Vars + Continuous_Vars + Noise_Vars, OCT)
    g.load_state_dict(torch.load("netG_epoch_036_iter_04.pth"))

    for i in [fe, d, q, g]:
        i.eval()
        i.cuda()

    trainer = Trainer(g, fe, d, q, Discrete_Vars, Continuous_Vars, Noise_Vars, 28, 28, 10, OCT)
    trainer.run()

if __name__ == "__main__":
    main()
