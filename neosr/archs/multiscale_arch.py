from torch import nn as nn
from torch.nn import functional as F
from .patchgan_arch import patchgan

from neosr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class multiscale(nn.Module):
    """Define a multi-scale discriminator, each discriminator is a instance of PatchDiscriminator.

    Args:
        num_layers (int or list): If the type of this variable is int, then degrade to PatchDiscriminator.
                                  If the type of this variable is list, then the length of the list is
                                  the number of discriminators.
        use_downscale (bool): Progressive downscale the input to feed into different discriminators.
                              If set to True, then the discriminators are usually the same.
    """

    def __init__(self,
                 num_in_ch,
                 num_feat=64,
                 num_layers=3,
                 max_nf_mult=8,
                 norm_type='batch',
                 use_sigmoid=False,
                 use_sn=False,
                 use_downscale=False):
        super(multiscale, self).__init__()

        if isinstance(num_layers, int):
            num_layers = [num_layers]

        # check whether the discriminators are the same
        if use_downscale:
            assert len(set(num_layers)) == 1
        self.use_downscale = use_downscale

        self.num_dis = len(num_layers)
        self.dis_list = nn.ModuleList()
        for nl in num_layers:
            self.dis_list.append(
                patchgan(
                    num_in_ch,
                    num_feat=num_feat,
                    num_layers=nl,
                    max_nf_mult=max_nf_mult,
                    norm_type=norm_type,
                    use_sigmoid=use_sigmoid,
                    use_sn=use_sn,
                ))

    def forward(self, x):
        outs = []
        h, w = x.size()[2:]

        y = x
        for i in range(self.num_dis):
            if i != 0 and self.use_downscale:
                y = F.interpolate(y, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
                h, w = y.size()[2:]
            outs.append(self.dis_list[i](y))

        return outs