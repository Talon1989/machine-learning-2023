import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn


#  LOADING MNIST DATASET

batch_size = 2**6

train_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    ),
    batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    ),
    batch_size=batch_size
)


#  CREATE MODEL

class RBM(nn.Module):

    def __init__(self, n_vis=784, n_hid=500, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(
            torch.sign(
                p - torch.autograd.Variable(
                    # torch.rand(p.size())
                    torch.distributions.Uniform(low=0, high=1).sample(p.size())
                )
            )
        )

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, initial_v):
        v_ = None
        pre_h1, h1 = self.v_to_h(initial_v)
        h_ = h1
        for _ in range(self.k):
            pre_v, v_ = self.h_to_v(h_)
            pre_h, h_ = self.v_to_h(v_)
        return initial_v, v_

    def free_energy(self, v):
        v_bias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (- hidden_term - v_bias_term).mean()


#  IMAGE VISUALIZATION METHOD

def show_images(img_1, img_2):

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(np.transpose(img_1.numpy(), axes=(1, 2, 0)))
    plt.title('sample image')

    plt.subplot(2, 1, 2)
    plt.imshow(np.transpose(img_2.numpy(), axes=(1, 2, 0)))
    plt.title('predicted image')

    plt.show()
    plt.clf()


def show(img):
    plt.imshow(np.transpose(img.numpy(), axes=(1, 2, 0)))
    plt.show()
    plt.clf()


#  TRAIN

epochs = 10
rbm = RBM(k=1)
optimizer = torch.optim.SGD(rbm.parameters(), lr=1/10)

for ep in range(1, epochs+1):
    losses = []
    v, v1 = None, None
    for _, (data, target) in enumerate(train_loader):
        data = torch.autograd.Variable(data.view(-1, 784))
        sample_data = data.bernoulli()
        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # show(make_grid(v1.view(32, 1, 28, 28).data))
    # show(make_grid(v1.view(v1.size(0), 1, 28, 28).data))
    show_images(
        make_grid(v.view(v.size(0), 1, 28, 28).data),
        make_grid(v1.view(v1.size(0), 1, 28, 28).data)
    )
    print('Epoch %d | Training loss : %.3f' % (ep, np.mean(losses)))









