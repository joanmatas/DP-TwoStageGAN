import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--large_num', required=True, help='num of generate data')
parser.add_argument('--model_path', default='../output/DP_WGANGP/netG_epoch_45.pth', help='Generator path')
parser.add_argument('--output_path', default='../data/generated_coarse', help='path of output')
opt = parser.parse_args()
large_num = int(opt.large_num)
ngf = 64
nc = 1
nz = 128
model_path = opt.model_path
output = opt.output_path
output_examples = '../data/generated_coarse_examples'
if not os.path.exists(output):
    os.mkdir(output)
if not os.path.exists(output_examples):
    os.mkdir(output_examples)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        # inputs is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = outputs.reshape((-1, 3, 32, 32))
        return outputs

netG = Generator()
netG = torch.load(model_path)

device = torch.device("cuda")
netG.to(device)

print(sum(param.numel() for param in netG.parameters()))

save_freq = 500
threshold = 15
for i in tqdm(range(large_num)):
    noise = torch.randn(1, 128, 1, 1, device=device)
    fake_imgs = netG(noise)
    fake_cpu = fake_imgs.cpu()
    fake_cpu = ((fake_cpu / 2) + 0.5) * 795
    path = os.path.join(output, str(i) + '.npy')
    traj_mat = fake_cpu[0].detach().numpy().astype(int)
    traj_mat[traj_mat < threshold] = 0
    np.save(path, traj_mat)
    if i % save_freq == 0:
        vutils.save_image(fake_cpu[0].detach(), f"{output_examples}/{str(i)}.png", normalize=False)
        vutils.save_image(fake_cpu[0].detach(), f"{output_examples}/{str(i)}_normalized.png", normalize=True)