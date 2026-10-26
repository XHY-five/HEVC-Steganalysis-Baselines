import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os


class CustomStem(nn.Module):
    """自定义Stem层，用于处理高分辨率输入和自定义通道数"""

    def __init__(self, in_channels, out_channels, orig_stem, stride=4):
        super().__init__()
        # 原始Stem参数
        kernel_size = orig_stem.kernel_size
        padding = orig_stem.padding

        # 创建新的Stem层
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,  # 使用更大的步长
            padding=padding,
            bias=(orig_stem.bias is not None)
        )

        # 初始化权重
        if orig_stem.weight is not None:
            self._initialize_weights(orig_stem.weight.data)
        else:
            nn.init.kaiming_normal_(
                self.conv.weight,
                mode='fan_out',
                nonlinearity='relu')

        # BatchNorm和Swish激活
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.swish = nn.SiLU(inplace=True)

    def _initialize_weights(self, orig_weight):
        """使用原始权重初始化新权重"""
        # 获取原始权重形状 [out_c, in_c, kH, kW]
        orig_out_c, orig_in_c, kH, kW = orig_weight.shape

        # 创建新权重张量 [out_c, new_in_c, kH, kW]
        new_weight = torch.zeros(
            orig_out_c,
            self.conv.in_channels,
            kH, kW
        ).to(orig_weight.device)

        # 复制原始权重到前min(orig_in_c, self.conv.in_channels)个通道
        copy_channels = min(orig_in_c, self.conv.in_channels)
        new_weight[:, :copy_channels] = orig_weight[:, :copy_channels]

        # 对于额外通道，使用原始权重的平均值
        if self.conv.in_channels > orig_in_c:
            mean_weight = orig_weight.mean(dim=1, keepdim=True)
            for i in range(orig_in_c, self.conv.in_channels):
                new_weight[:, i] = mean_weight.squeeze(1)

        self.conv.weight.data = new_weight

        # 初始化偏置（如果有）
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.swish(x)


class DualBranchEnsemble(nn.Module):
    def __init__(self):
        super().__init__()

        # === IPM 分支：EfficientNet-B3 + 预训练 ===
        self.ipm_branch = self._build_branch('efficientnet-b3', in_channels=37)

        # === Coeff 分支：EfficientNet-B1 + dilation + 预训练 ===
        self.coeff_branch = self._build_branch(
            'efficientnet-b1', in_channels=67)
        self.apply_dilation_to_effnet_branch(self.coeff_branch, dilation=8)

        # === 梯度检查点 ===
        self.ipm_branch.set_swish(memory_efficient=True)
        self.coeff_branch.set_swish(memory_efficient=True)

        # === 特征融合 ===
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion = nn.Sequential(
            nn.Linear(1536 + 1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def _build_branch(self, model_name, in_channels):
        """构建带有自定义Stem的分支"""
        # 创建基础模型（不加载权重）
        model = EfficientNet.from_name(model_name)

        # 保存原始Stem参数
        orig_stem = model._conv_stem
        orig_out_channels = orig_stem.out_channels

        # 创建自定义Stem
        custom_stem = CustomStem(
            in_channels=in_channels,
            out_channels=orig_out_channels,
            orig_stem=orig_stem,
            stride=4  # 使用更大的步长处理高分辨率
        )

        # 替换原始Stem
        model._conv_stem = custom_stem

        # 加载预训练权重（跳过Stem）
        self._load_pretrained_except_stem(model, model_name)

        return model

    def _load_pretrained_except_stem(self, model, model_name):
        """加载本地预训练权重，跳过Stem层"""
        try:
            # 1. 指定本地权重文件路径
            ckpt_map = {
                'efficientnet-b3': 'efficientnet_path/efficientnet-b3-5fb5a3c3.pth',
                'efficientnet-b1': 'efficientnet_path/efficientnet-b1-f1951068.pth'
            }
            ckpt_path = ckpt_map.get(model_name)
            assert ckpt_path is not None, f"Unknown model name: {model_name}"
            assert os.path.exists(
                ckpt_path), f"Checkpoint not found: {ckpt_path}"

            # 2. 加载预训练权重
            pretrained_dict = torch.load(ckpt_path, map_location='cpu')

            # 3. 过滤掉 stem 层的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if not k.startswith('_conv_stem')}

            # 4. 加载
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print(
                f"Loaded pretrained weights from {ckpt_path} (excluding stem)")

        except Exception as e:
            print(f"Failed to load pretrained weights from local file: {e}")
            print("Using random initialization for all layers")

    def apply_dilation_to_effnet_branch(self, model, dilation=8):
        total_blocks = len(model._blocks)
        for i, block in enumerate(model._blocks):
            if i < total_blocks - 2:  # 只改最后两层
                continue
            if hasattr(
                    block, '_depthwise_conv') and block._depthwise_conv.stride == (1, 1):
                block._depthwise_conv.dilation = (dilation, dilation)
                block._depthwise_conv.padding = (dilation, dilation)
                block._depthwise_conv.stride = (1, 1)
                block.id_skip = False
                if hasattr(block, '_block_args'):
                    block._block_args = block._block_args._replace(
                        id_skip=False)

    def forward(self, ipm_input, coeff_input):
        # IPM分支
        ipm_feat = self.ipm_branch.extract_features(ipm_input)
        ipm_feat = self.pool(ipm_feat).flatten(1)
        # print("Coeff input shape:", coeff_input.shape)
        # Coeff分支
        # Coeff分支（不拆通道，直接36通道输入）
        coeff_feat = self.coeff_branch.extract_features(coeff_input)

        coeff_feat = self.pool(coeff_feat).flatten(1)

        # 特征融合
        x = torch.cat([ipm_feat, coeff_feat], dim=1)
        return self.fusion(x)
