import torch
from torch import nn
import torch.nn.functional as F

ONNX_TO_ENCODER = {
    "Conv_40": "conv1",
    "Conv_42": "conv2",
    "Conv_44": "conv3",
    "Conv_46": "conv4",
}

ONNX_TO_DECODER = {
    "Conv_590": "conv1",
    "Conv_594": "conv2",
    "Conv_596": "conv3",
    "Conv_612": "conv4",
}

ONNX_TO_CONV_BLOCKS = {
    "Conv_62": (0, "conv1"),
    "Conv_107": (0, "conv2"),
    "Conv_152": (1, "conv1"),
    "Conv_197": (1, "conv2"),
    "Conv_242": (2, "conv1"),
    "Conv_287": (2, "conv2"),
    "Conv_332": (3, "conv1"),
    "Conv_377": (3, "conv2"),
    "Conv_422": (4, "conv1"),
    "Conv_467": (4, "conv2"),
    "Conv_512": (5, "conv1"),
    "Conv_557": (5, "conv2"),
    "Gemm_73": (0, "condition_embed_1"),
    "Gemm_118": (0, "condition_embed_2"),
    "Gemm_163": (1, "condition_embed_1"),
    "Gemm_208": (1, "condition_embed_2"),
    "Gemm_253": (2, "condition_embed_1"),
    "Gemm_298": (2, "condition_embed_2"),
    "Gemm_343": (3, "condition_embed_1"),
    "Gemm_388": (3, "condition_embed_2"),
    "Gemm_433": (4, "condition_embed_1"),
    "Gemm_478": (4, "condition_embed_2"),
    "Gemm_523": (5, "condition_embed_1"),
    "Gemm_568": (5, "condition_embed_2"),
}


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        x: NxCxHxW
        condition: Nx2C
        """
        B, C, _, _ = x.size()

        if condition.size(1) != 2 * C:
            raise ValueError(
                f"AdaptiveInstanceNorm requires inputs of shape x = (B, C, H, W) and condition = (B, 2 * C), but 2 * {x.size(1)} does not match {cond.size(1)}."
            )

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, correction=0)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        scale, shift = condition.chunk(2, dim=1)
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)

        out = x_norm * scale + shift

        return out


class AdaptiveConvBlock(nn.Module):
    def __init__(
        self, num_channels: int = 1024, condition_dim: int = 512, eps: float = 1e-8
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            padding_mode="reflect",
        )

        self.condition_embed_1 = nn.Linear(
            in_features=condition_dim, out_features=2 * num_channels, bias=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            padding_mode="reflect",
        )

        self.condition_embed_2 = nn.Linear(
            in_features=condition_dim, out_features=2 * num_channels, bias=True
        )

        self.norm = AdaptiveInstanceNorm(eps=eps)
        self.relu = nn.ReLU()

    def forward(self, x, condition):
        residual = x

        x = self.conv1(x)

        cond_embed = self.condition_embed_1(condition)
        x = self.norm(x=x, condition=cond_embed)

        x = self.relu(x)

        x = self.conv2(x)

        cond_embed = self.condition_embed_2(condition)
        x = self.norm(x=x, condition=cond_embed)

        out = residual + x
        return out


class InswapperEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            3,
            128,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3),
            padding_mode="reflect",
        )
        self.conv2 = torch.nn.Conv2d(
            128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = torch.nn.Conv2d(
            256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.conv4 = torch.nn.Conv2d(
            512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.conv_layers = nn.ModuleList(
            [self.conv1, self.conv2, self.conv3, self.conv4]
        )

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Args:
            x: tensor representing target image, with shape (B, C, H, W) (for inswapper_128, 1 x 3 x 128 x 128)

        Outputs:
            tensor with shape (B, C, H, W) (for inswapper_128, 1 x 1024 x 32 x 32)
        """

        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)

        return x


class InswapperDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            128,
            3,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3),
            padding_mode="reflect",
        )

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape B x C x H x W (for inswapper_128, 1024 x 32 x 32)

        Outputs:
            image-sized tensor of shape B x C x H x W (for inswapper_128, 1 x 3 x 128 x 128)
        """
        # Upsample
        x = F.interpolate(x, scale_factor=[2, 2], mode="bilinear")

        x = self.conv1(x)
        x = self.leaky_relu(x)

        # Upsample
        x = F.interpolate(x, scale_factor=[2, 2], mode="bilinear")

        x = self.conv2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = (self.tanh(x) + 1) / 2  # clamp to [0, 1]

        return x


class InswapperModel(nn.Module):
    def __init__(
        self,
        num_blocks: int = 6,
        num_channels: int = 1024,
        condition_dim: int = 512,
        eps: float = 1e-8,
        onnx_checkpoint: str | None = None,
    ):
        super().__init__()

        self.encoder = InswapperEncoder()

        conv_blocks = []

        for _ in range(num_blocks):
            block = AdaptiveConvBlock(
                num_channels=num_channels, condition_dim=condition_dim, eps=eps
            )
            conv_blocks.append(block)

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.decoder = InswapperDecoder()

        # load weights if ONNX checkpoint is provided
        if onnx_checkpoint is not None:
            self._load_from_onnx_checkpoint(onnx_checkpoint)

    def _load_from_onnx_checkpoint(self, checkpoint: str):
        from onnx2torch import convert

        with torch.no_grad():
            model = convert(checkpoint)

            for name, module in model.named_modules():
                if isinstance(module, torch.fx.GraphModule):
                    continue

                if name == "initializers":
                    continue

                if name in ONNX_TO_ENCODER:
                    own_module = getattr(self.encoder, ONNX_TO_ENCODER[name])
                    if hasattr(own_module, "weight"):
                        own_module.weight.copy_(getattr(model, name).weight)
                    if hasattr(own_module, "bias"):
                        own_module.bias.copy_(getattr(model, name).bias)
                elif name in ONNX_TO_DECODER:
                    own_module = getattr(self.decoder, ONNX_TO_DECODER[name])
                    if hasattr(own_module, "weight"):
                        own_module.weight.copy_(getattr(model, name).weight)
                    if hasattr(own_module, "bias"):
                        own_module.bias.copy_(getattr(model, name).bias)
                elif name in ONNX_TO_CONV_BLOCKS:
                    block_i, own_name = ONNX_TO_CONV_BLOCKS[name]
                    own_module = getattr(self.conv_blocks[block_i], own_name)
                    if hasattr(own_module, "weight"):
                        own_module.weight.copy_(getattr(model, name).weight)
                    if hasattr(own_module, "bias"):
                        own_module.bias.copy_(getattr(model, name).bias)

    def forward(self, target_image, source_face):
        # encoder
        x = self.encoder(target_image)

        # conv blocks
        for block in self.conv_blocks:
            x = block(x, source_face)

        # decoder
        out = self.decoder(x)
        return out


if __name__ == "__main__":
    model = InswapperModel(onnx_checkpoint="models/inswapper_128.onnx")
    print(model)
