import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:
    def __init__(self, G, FE, D, Q, Discrete_Vars, Continuous_Vars):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q

        self.Discrete_Vars = Discrete_Vars
        self.Continuous_Vars = Continuous_Vars

        self.Continuous_Steps = 12

        self.batch_size = self.Discrete_Vars * self.Continuous_Steps
        self.Image_Width = 28
        self.Image_Height = 28

    def _noise_sample(self, dis_c, con_c, noise, bs):
        idx = np.random.randint(self.Discrete_Vars, size=bs)
        c = np.zeros((bs, self.Discrete_Vars))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

        return z, idx

    def train(self):
        real_x = torch.FloatTensor(self.batch_size, 1, self.Image_Width, self.Image_Height).cuda()
        label = torch.FloatTensor(self.batch_size, 1).cuda()
        dis_c = torch.FloatTensor(self.batch_size, self.Discrete_Vars).cuda()
        con_c = torch.FloatTensor(self.batch_size, self.Continuous_Vars).cuda()
        noise = torch.FloatTensor(self.batch_size, 62).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {
                            'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()}, {
                            'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

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
        one_hot = np.zeros((self.batch_size, self.Discrete_Vars))
        one_hot[range(self.batch_size), idx] = 1
        fix_noise = torch.Tensor(self.batch_size, 62).uniform_(-1, 1)

        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):
                # real part
                optimD.zero_grad()

                x, _ = batch_data

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, self.Discrete_Vars)
                con_c.data.resize_(bs, self.Continuous_Vars)
                noise.data.resize_(bs, 62)

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

                if num_iters % 100 == 0:

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    for i in range(self.Continuous_Vars):
                        con_c.data.copy_(torch.from_numpy(continuous_vars[i]))
                        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                        x_save = self.G(z)

                        #NOTE: nrow is actually images PER ROW! NOT the number of rows!
                        save_image(x_save.data, './tmp/{:03d}_{:02d}_c{:02d}.png'.format(epoch, num_iters // 100, i),
                            nrow=self.Continuous_Steps)
