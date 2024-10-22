import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    EfficientNet_B7_Weights,
    Inception_V3_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    efficientnet_b7,
    inception_v3,
    resnet101,
    vgg,
)

from neosr.utils.registry import LOSS_REGISTRY


class VGG(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg_pretrained_features = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        vgg_pretrained_features.eval()

        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()

        # vgg19
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 128, 256, 512, 512]

    def get_features(self, x):
        # normalize the data
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h

        h = self.stage2(h)
        h_relu2_2 = h

        h = self.stage3(h)
        h_relu3_3 = h

        h = self.stage4(h)
        h_relu4_3 = h

        h = self.stage5(h)
        h_relu5_3 = h

        # get the features of each layer
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x):
        return self.get_features(x)


class ResNet(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        model.eval()

        self.stage1 = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.stage2 = nn.Sequential(model.maxpool, model.layer1)
        self.stage3 = nn.Sequential(model.layer2)
        self.stage4 = nn.Sequential(model.layer3)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 256, 512, 1024]  #

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]  #

    def forward(self, x):
        return self.get_features(x)


class Inception(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        self.stage1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3
        )
        self.stage2 = nn.Sequential(
            inception.maxpool1, inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3
        )
        self.stage3 = nn.Sequential(
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
        )
        self.stage4 = nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.chns = [64, 192, 288, 768]

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

    def get_features(self, x):
        h = (x - self.mean) / self.std
        # h = (x-0.5)*2
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

    def forward(self, x):
        return self.get_features(x)


class EffNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT
        ).features  # [:6]
        model.eval()

        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False
        self.chns = [32, 48, 80, 160, 224]

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]  #

    def forward(self, x):
        return self.get_features(x)


@LOSS_REGISTRY.register()
class fdl_loss(nn.Module):
    def __init__(
        self,
        patch_size=5,
        stride=1,
        num_proj=24,
        model="vgg",
        phase_weight=1.0,
        loss_weight=1.0,
    ):
        """
        Adapted from: https://github.com/eezkni/FDL

        Args:
        ---
            patch_size (int), stride (int), num_proj (int): SWD slice parameters.
            model (str): feature extractor, supports VGG, ResNet, Inception, EffNet.
            phase_weight (float): weight for phase branch.
            loss_weight (float): loss weight.
        """

        super().__init__()
        model = model.lower()

        if model == "resnet":
            self.model = ResNet()
        elif model == "effnet":
            self.model = EffNet()
        elif model == "inception":
            self.model = Inception()
        elif model == "vgg":
            self.model = VGG()
        else:
            msg = "Invalid model type! Valid models: VGG, Inception, EffNet, ResNet"
            raise NotImplementedError(msg)

        self.phase_weight = phase_weight
        self.loss_weight = loss_weight
        self.stride = stride
        for i in range(len(self.model.chns)):
            rand = torch.randn(
                num_proj, self.model.chns[i], patch_size, patch_size, device="cuda"
            )
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)

    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = getattr(self, f"rand_{idx}")
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        return torch.abs(projx - projy).mean([1, 2])

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        score = []
        for i in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[i], dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            s_amplitude = self.forward_once(x_mag, y_mag, i)
            s_phase = self.forward_once(x_phase, y_phase, i)

            score.append(s_amplitude + s_phase * self.phase_weight)

        score = sum(score)
        # decrease magnitude to balance with other losses
        score = score.mean() * 0.01

        return score * self.loss_weight
