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


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)

        return logli.sum(1).mean().mul(-1)


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

        self.Continuous_Steps = 4

        self.Noise_Vars = Noise_Vars

        self.batch_size = batch_size
        self.Progress_Batch_Size = self.Discrete_Vars * self.Continuous_Steps
        self.Image_Width = Image_Width
        self.Image_Height = Image_Height

        self.OCT = OCT

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(self.Discrete_Vars, size=bs)
        c = np.zeros((bs, self.Discrete_Vars))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.Total_Vars, 1, 1)

        return z, idx

    def train(self):
        real_x_tensor = torch.FloatTensor(self.batch_size, 1, self.Image_Width, self.Image_Height).cuda()
        label_tensor = torch.FloatTensor(self.batch_size, 1).cuda()
        dis_c_tensor = torch.FloatTensor(self.batch_size, self.Discrete_Vars).cuda()
        con_c_tensor = torch.FloatTensor(self.batch_size, self.Continuous_Vars).cuda()
        noise_tensor = torch.FloatTensor(self.batch_size, self.Noise_Vars).cuda()

        real_x = Variable(real_x_tensor)
        label = Variable(label_tensor, requires_grad=False)
        dis_c = Variable(dis_c_tensor)
        con_c = Variable(con_c_tensor)
        noise = Variable(noise_tensor)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {
                            'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()}, {
                            'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

        if self.OCT:
            OCT_images = np.load("./A/np/train/all.npy") / (1<<16)
            dataset = TensorDataset(torch.Tensor(OCT_images))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        else:
            dataset = dset.MNIST(
                './dataset', transform=transforms.ToTensor(), download=True)
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)


        # fixed random variables
        c = np.linspace(-1, 1, self.Continuous_Steps).reshape(1, -1)
        c = np.repeat(c, self.Discrete_Vars, 0).reshape(-1, 1)

        continuous_vars = []
        for i in range(self.Continuous_Vars):
            continuous_vars.append(np.hstack([np.zeros_like(c) for j in range(i)] +
                [c] + [np.zeros_like(c) for j in range(self.Continuous_Vars - i - 1)]))

        idx = np.arange(self.Discrete_Vars).repeat(self.Continuous_Steps)
        one_hot = np.zeros((self.Progress_Batch_Size, self.Discrete_Vars))
        one_hot[range(self.Progress_Batch_Size), idx] = 1
        fix_noise = torch.Tensor(self.Progress_Batch_Size, self.Noise_Vars).uniform_(-1, 1)
        
        dis_c_progress = Variable(dis_c_tensor)
        con_c_progress = Variable(con_c_tensor)
        noise_progress = Variable(noise_tensor)

        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):
                # real part
                optimD.zero_grad()

                if self.OCT:
                    x = batch_data[0]
                else:
                    x, _ = batch_data

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, self.Discrete_Vars)
                con_c.data.resize_(bs, self.Continuous_Vars)
                noise.data.resize_(bs, self.Noise_Vars)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters == 0 and epoch % 5 == 0:

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )
                    dis_c_progress.data.resize_(self.Progress_Batch_Size, self.Discrete_Vars)
                    con_c_progress.data.resize_(self.Progress_Batch_Size, self.Continuous_Vars)
                    noise_progress.data.resize_(self.Progress_Batch_Size, self.Noise_Vars)
                    noise_progress.data.copy_(fix_noise)
                    dis_c_progress.data.copy_(torch.Tensor(one_hot))

                    for i in range(self.Continuous_Vars):
                        con_c.data.copy_(torch.from_numpy(continuous_vars[i]))
                        z = torch.cat([noise_progress, dis_c_progress, con_c_progress], 1).view(-1, self.Total_Vars, 1, 1)
                        x_save = self.G(z)

                        #NOTE: nrow is actually images PER ROW! NOT the number of rows!
                        save_image(x_save.data, './tmp/{:03d}_{:02d}_c{:02d}.png'.format(epoch, num_iters // 100, i),
                            nrow=self.Continuous_Steps)
                        #save_single_image(x[0], './tmp/{:03d}_{:02d}_first.png'.format(epoch, num_iters // 100, i))
                        torch.save(self.G.state_dict(), './netG_epoch_%d.pth' % (epoch))
                        torch.save(self.D.state_dict(), './netD_epoch_%d.pth' % (epoch))
                        torch.save(self.Q.state_dict(), './netQ_epoch_%d.pth' % (epoch))
                        torch.save(self.FE.state_dict(), './netFE_epoch_%d.pth' % (epoch))
            print("Epoch: {0}".format(epoch))
