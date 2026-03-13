import torch
import torch.nn as nn


class Generator_Loss(nn.Module):
    """Generator loss: distance + adversarial + perceptual."""

    def __init__(self, Generator_config):
        super(Generator_Loss, self).__init__()
        self.Loss_adv_weight = Generator_config['Loss_adv_weight']
        self.Loss_Dist_weight = Generator_config['Loss_Dist_weight']
        self.Loss_percep_weight = Generator_config.get('Loss_percep_weight', 1.0)
        self.Dist_Loss = Generator_config['Dist_Loss']

        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.perceptual_loss = nn.L1Loss()

        from torchvision import models

        vgg = models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, Input, Generator, Discriminator, confidence):
        # Pixel/feature distance to source images
        loss_dist = 0
        for pair in self.Dist_Loss:
            loss_dist = loss_dist + self.MSELoss(Generator[pair[0]], Input[pair[1]])

        # Adversarial target for generator must be "real" (ones)
        loss_adv = 0
        for name in Discriminator:
            pred = Discriminator[name]
            target = torch.ones_like(pred)
            loss_adv = loss_adv + self.BCELoss(pred, target)

        # Perceptual loss (VGG frozen, but gradients flow to Generator)
        self.vgg = self.vgg.to(Input['Vis'].device)
        feat_real = self.vgg(Input['Vis'])
        feat_fake = self.vgg(Generator['Generator_1'])
        loss_percep = self.perceptual_loss(feat_real, feat_fake)

        return (
            self.Loss_Dist_weight * loss_dist
            + self.Loss_adv_weight * loss_adv
            + self.Loss_percep_weight * loss_percep
        )


class Discriminator_Loss(nn.Module):
    """Discriminator loss."""

    def __init__(self, Discriminator_config):
        super(Discriminator_Loss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, Generator, Discriminator_output, confidence):
        loss_adv = 0
        for name in Discriminator_output:
            pred = Discriminator_output[name]
            target = confidence[name]
            while target.ndim < pred.ndim:
                target = target.unsqueeze(-1)
            target = target.expand_as(pred)
            loss_adv += self.BCELoss(pred, target)
        return loss_adv
