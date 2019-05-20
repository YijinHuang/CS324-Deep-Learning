import argparse
import os
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

n_epochs = 100
batch_size = 64
lr = 0.0002
dimensionality = 100
save_interval = 500


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel):
        super(Generator, self).__init__()

        channel_rate = [8, 4, 2]

        self.generator = nn.Sequential(
            self.basic_block(in_channels, base_channel * channel_rate[0], 4, 1, 0),
            self.basic_block(base_channel * channel_rate[0], base_channel * channel_rate[1], 4, 2, 1),
            self.basic_block(base_channel * channel_rate[1], base_channel * channel_rate[2], 4, 2, 1),
            self.basic_block(base_channel * channel_rate[2], base_channel, 4, 2, 1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, 1, 0, dilation=2, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        x = self.generator(z)
        x = self.refine(x)
        return x

    def basic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channel):
        super(Discriminator, self).__init__()

        channel_rate = [1, 2, 4, 8]

        self.discriminator = nn.Sequential(
            self.basic_block(in_channels, base_channel * channel_rate[0], 4, 2, 1),
            self.basic_block(base_channel * channel_rate[0], base_channel * channel_rate[1], 4, 2, 1),
            self.basic_block(base_channel * channel_rate[1], base_channel * channel_rate[2], 4, 2, 1),
            self.basic_block(base_channel * channel_rate[2], base_channel * channel_rate[3], 3, 1, 0)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(base_channel * channel_rate[3], 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        x = self.discriminator(img)
        x = self.predict(x).squeeze()
        return x

    def basic_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    record_epochs = []
    dis_losses = []
    gen_losses = []

    cross_entropy = nn.BCELoss()

    generator = generator.cuda()
    discriminator = discriminator.cuda()
    for epoch in range(n_epochs):
        epoch_dis_loss = 0
        epoch_gen_loss = 0

        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            fake_y = torch.zeros(batch_size).cuda()
            true_y = torch.ones(batch_size).cuda()

            # Train Discriminator
            # -------------------
            discriminator.zero_grad()

            true_pred = discriminator(imgs)
            true_loss = cross_entropy(true_pred, true_y)

            noise = torch.randn((batch_size, dimensionality)).view(batch_size, dimensionality, 1, 1).cuda()
            gen_img = generator(noise)
            fake_pred = discriminator(gen_img.detach())
            fake_loss = cross_entropy(fake_pred, fake_y)

            loss = true_loss + fake_loss
            epoch_dis_loss += loss

            loss.backward()
            optimizer_D.step()

            # Train Generator
            # ---------------
            generator.zero_grad()

            fake_pred = discriminator(gen_img)
            gen_loss = cross_entropy(fake_pred, true_y)
            epoch_gen_loss += gen_loss

            gen_loss.backward()
            optimizer_G.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_img[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
                torch.save(generator, './models/G_{}'.format(batches_done))
                torch.save(discriminator, './models/D_{}'.format(batches_done))

        # Record
        # ---------------
        dis_losses.append(epoch_dis_loss/(i+1))
        gen_losses.append(gen_loss/(i+1))
        record_epochs.append(epoch)
        print('Epoch: {}, Generator loss: {}, Discriminator loss: {}'.format(record_epochs[-1], gen_losses[-1], dis_losses[-1]))

    return record_epochs, dis_losses, gen_losses


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize models and optimizers
    generator = Generator(100, 1, 128)
    discriminator = Discriminator(1, 128)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Start training
    record = train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    pickle.dump(record, open('./record', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
