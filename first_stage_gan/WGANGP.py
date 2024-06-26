import argparse
import os
import random
import numpy as np
import functools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", default="../data/grid32_train", help="path to dataset")
parser.add_argument("--labelroot", default="../data/traj_all.txt", help="path to label")
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=4
)
parser.add_argument("--batchSize", type=int, default=64, help="inputs batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=32,
    help="the height / width of the inputs image to network",
)
parser.add_argument("--nz", type=int, default=128, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument(
    "--niter", type=int, default=100, help="number of epochs to train for"
)
parser.add_argument(
    "--n_critic",
    type=int,
    default=1,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=4, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument(
    "--outf", default="../output/WGANGP", help="folder to output images and model checkpoints"
)
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument("--start_iter", type=int, default=0)

opt = parser.parse_args()
print(opt)
if opt.netG != "" and opt.start_iter == 0:
    print("start_iter is wrong!!!!")
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if opt.cuda else "cpu")
ngpu = torch.cuda.device_count()
# ngpu = 1
nz = int(opt.nz)
ndf = int(opt.ndf)
ngf = int(opt.ngf)
nc = 3
# Loss weight for gradient penalty
lambda_gp = 10

generator_losses = []
discriminator_losses = []
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="Generator Loss")
ax1 = fig.add_subplot(122, title="Discriminator Loss")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, generator_losses, 'bo-')
    ax1.plot(x_epoch, discriminator_losses, 'ro-')
    fig.savefig('train_test.jpg')

def default_loader(path):
    return np.load(path)

class myImageFolder(data.Dataset):
    def __init__(
        self,
        root,
        label_root,
        target_transform=None,
        transform=None,
        loader=default_loader,
    ):
        fh = open(label_root)
        img_names = []
        for line in fh.readlines():
            cls = line.split()
            fn = cls.pop(0)
            fn = fn[:-1]
            if os.path.isfile(os.path.join(root, fn)):
                img_names.append(fn)
        self.imgs = img_names
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform:
            img = self.gen_sample(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def gen_sample(self, img):
        img = img.astype(float)
        img = img.transpose((1, 2, 0))
        img /= 795.0
        img -= [0.5, 0.5, 0.5]  # [0.172185785,0.033457624, 0.003309033]
        img /= [0.5, 0.5, 0.5]
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img)
        return img


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.InstanceNorm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.main = nn.Sequential(
            # inputs is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            self.InstanceNorm(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            self.InstanceNorm(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            self.InstanceNorm(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = outputs.reshape((-1, 3, 32, 32))
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.InstanceNorm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 1, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
            self.InstanceNorm(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            self.InstanceNorm(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, inputs):
        outputs = self.main(inputs)

        return outputs.view(-1, 1).squeeze(1)


def calculate_gradient_penalty(netD, real_imgs, fake_imgs):
    """Calculates the gradient penalty loss for WGAN GP"""
    eta = torch.FloatTensor(real_imgs.size(0), 1, 1, 1).uniform_(0, 1).to(device)
    eta = eta.expand(
        real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)
    ).to(device)

    interpolated = eta * real_imgs + ((1 - eta) * fake_imgs)
    interpolated.to(device)

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probaility of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * 10

    return gradients_penalty


def train(dataloader, optimizerD, optimizerG, netD, netG):
    writer = SummaryWriter(os.path.join(opt.outf, "log"))
    for epoch in range(opt.start_iter, opt.niter):
        for i, (real_imgs) in enumerate(dataloader):
            real_imgs = real_imgs.float()

            # configure input
            real_imgs = real_imgs.to(device)

            # Get real imgs batch size
            batch_size = real_imgs.size(0)

            # -----------------
            #  Train Discriminator
            # -----------------

            netD.zero_grad()

            # Sample noise as generator input
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            # Generate a batch of images
            fake_imgs = netG(noise)

            # Real images
            real_validity = netD(real_imgs)
            # Fake images
            fake_validity = netD(fake_imgs)
            # Gradient penalty
            gradient_penalty = calculate_gradient_penalty(
                netD, real_imgs.data, fake_imgs.data
            )

            # Loss measures generator's ability to fool the discriminator
            errD = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

            errD.backward(retain_graph=True)
            optimizerD.step()

            optimizerG.zero_grad()

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # ---------------------
                #  Train Generator
                # ---------------------

                # Generate a batch of images
                fake_imgs = netG(noise)
                # Adversarial loss
                errG = -torch.mean(netD(fake_imgs))

                errG.backward(retain_graph=True)
                optimizerG.step()
            
            # losses visualization
            if i == 0:
                generator_losses.append(errG.item())
                discriminator_losses.append(errD.item())
                draw_curve(epoch)

            if i % 100 == 0:
                print(
                    f"[{epoch + 1}/{opt.niter}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} "
                    f"Loss_G: {errG.item():.4f}."
                )
                time_step = epoch * int(len(dataloader) / 100) + int(i / 100)
                writer.add_scalar("loss_D", errD.item(), time_step)
                writer.add_scalar("loss_G", errG.item(), time_step)
                writer.add_scalar("gradients_penalty", gradient_penalty.item(), time_step)
                
            if epoch % 1 == 0:
                vutils.save_image(
                    (real_imgs / 2 + 0.5) * 795,
                    f"{opt.outf}/real_samples_normalized.png",
                    normalize=True,
                )
                vutils.save_image(
                    (real_imgs / 2 + 0.5) * 795,
                    f"{opt.outf}/real_samples.png",
                    normalize=False,
                )
                vutils.save_image(
                    (netG(noise).detach() / 2 + 0.5) * 795,
                    f"{opt.outf}/fake_samples_epoch_{epoch}_normalized.png",
                    normalize=True,
                )
                vutils.save_image(
                    (netG(noise).detach() / 2 + 0.5) * 795,
                    f"{opt.outf}/fake_samples_epoch_{epoch}.png",
                    normalize=False,
                )

            # do checkpointing

        torch.save(netG, f"{opt.outf}/netG_epoch_{epoch + 1}.pth")
        torch.save(netD, f"{opt.outf}/netD_epoch_{epoch + 1}.pth")


def main():
    dataset = myImageFolder(
        root=opt.dataroot,
        label_root=opt.labelroot,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    assert dataset
    print("Loading data")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize * ngpu,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True,
    )
    print("Preparing Model")
    netG = Generator()
    netD = Discriminator()
    if opt.cuda:
        netD.cuda()
        netG.cuda()
    if ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        netG = nn.DataParallel(netG, device_ids=[0, 1, 2, 3])
        netD = nn.DataParallel(netD, device_ids=[0, 1, 2, 3])

    netG.apply(weights_init)
    if opt.netG != "":
        state_dict = torch.load(opt.netG).state_dict()
        netG.load_state_dict(state_dict)
    netD.apply(weights_init)
    if opt.netD != "":
        state_dict = torch.load(opt.netD).state_dict()
        netD.load_state_dict(state_dict)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    print("Training")
    train(dataloader, optimizerD, optimizerG, netD, netG)


if __name__ == "__main__":
    main()