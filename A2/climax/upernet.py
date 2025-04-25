import torch
from torch import nn, Tensor
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import math

class ConvModule(nn.Sequential):

    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(nn.Conv2d(c1, c2, k, s, p, d, g, bias=False), nn.GroupNorm(32, c2), nn.ReLU(True))


class PPM(nn.Module):
    __doc__ = 'Pyramid Pooling Module in PSPNet\n    '

    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvModule(c1, c2, 1)) for scale in scales])
        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate((stage(x)), size=(x.shape[-2:]), mode='bilinear', align_corners=True))
        else:
            outs = [
             x] + outs[::-1]
            out = self.bottleneck(torch.cat(outs, dim=1))
            return out

class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output

# based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/modeling_upernet.py#L222
class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, in_channels, channels, num_convs, concat_input, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1, 
        dropout_ratio: float = 0.1, num_classes: int = 19
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)

        self.classifier = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02) #arbitrary init with 0.02 her ebased on quick search on google
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.classifier(output)
        return output


class UPerHead(nn.Module):
    __doc__ = 'Unified Perceptual Parsing for Scene Understanding\n    https://arxiv.org/abs/1807.10221\n    scales: Pooling scales used in PPM module applied on the last feature\n    '

    def __init__(self, in_channels, channel=128, num_classes=19, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channel, scales)
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        for in_ch in in_channels[:-1]:
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))
        else:
            self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
            self.dropout = nn.Dropout2d(0.1)
            self.conv_seg = nn.Conv2d(channel, num_classes, 1)        

    def forward(self, features: Tuple[(Tensor, Tensor, Tensor, Tensor)]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]
        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=(feature.shape[-2:]), mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))
        else:
            fpn_features.reverse()
            for i in range(1, len(features)):
                fpn_features[i] = F.interpolate((fpn_features[i]), size=(fpn_features[0].shape[-2:]), mode='bilinear', align_corners=False)
            else:
                output = self.bottleneck(torch.cat(fpn_features, dim=1))
                output = self.conv_seg(self.dropout(output))
                return output


class UperNetNeck(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int):
        """
        Feature processing neck for UperNet.

        Args:
            in_channels (List[int]): List of input channels for each feature level.
            out_channels (int): Number of output channels for each feature level.
        """
        super().__init__()
        
        # Ensure in_channels is a list, and create separate conv layers for each feature level
        self.convs = nn.ModuleList([ConvModule(in_ch, out_channels, 3, 1, 1) for in_ch in in_channels])

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, ...]:
        """
        Processes feature maps from the backbone.

        Args:
            features (List[Tensor]): List of feature maps from the backbone.

        Returns:
            Tuple[Tensor, ...]: Processed feature maps for further use in UPerHead.
        """
        return tuple(conv(f) for conv, f in zip(self.convs, features))


class UperNet(nn.Module):

    def __init__(self, aux_head=None, in_channels=768, pool_scales=(1, 2, 3, 6), channels=512, num_classes=21, input_size=128):
        super().__init__()
        self.input_size = (input_size, input_size)

        # Compute the spatial size from the number of tokens
        self.patch_size = 16  # Assuming ViT outputs 16x16 tokens
        self.num_tokens = (input_size // self.patch_size) ** 2  # Should match 256 tokens

        self.neck = UperNetNeck(in_channels=[in_channels], out_channels=128)  # Adjusted for ViT
        self.decode_head = UPerHead(
            in_channels=[128],  # Ensure compatibility with UperNetNeck output
            scales=pool_scales,
            channel=channels,
            num_classes=num_classes
        )

        self.aux_head = aux_head

    def forward(self, x):
        print(f'Original input shape: {x.shape}')  # (1, 256, 768)

        batch_size, num_tokens, embedding_dim = x.shape
        height = width = int(math.sqrt(num_tokens))  # Assuming square spatial grid (16x16)

        # Reshape tokens into feature maps (B, C, H, W)
        x = x.permute(0, 2, 1).view(batch_size, embedding_dim, height, width)
        print(f'Reshaped to feature map: {x.shape}')  # (1, 768, 16, 16)

        # Pass through UperNetNeck
        features = self.neck([x])  # Wrap in a list since `UperNetNeck` expects multiple feature maps
        print(f'After neck features shape: {[f.shape for f in features]}')

        # Decode features
        x = self.decode_head(features)
        print(f'After decode head shape: {x.shape}')

        # Upsample to input size
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        if self.aux_head is not None:
            auxout = self.aux_head(features)
            auxout = F.interpolate(auxout, size=self.input_size, mode='bilinear', align_corners=False)
            return x, auxout
        else:
            return x
