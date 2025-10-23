import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn import tasks
from collections import OrderedDict

# ===================================================================
# PART 1: DEFINE THE CUSTOM MODULES
# ===================================================================

class SpecialConv(nn.Module):
    """
    Implementation of the Special Convolutional Layer from the MSDC-Net paper.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SpecialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attn_mean = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0), nn.Sigmoid())
        self.attn_mean_std = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0), nn.Sigmoid())

    def forward(self, x):
        features = self.conv(x)
        with torch.no_grad():
            b, c, h, w = x.shape
            mean = torch.mean(x.view(b, c, -1), dim=2).view(b, c, 1, 1)
            std = torch.std(x.view(b, c, -1), dim=2).view(b, c, 1, 1)
        attn1 = self.attn_mean(mean)
        mean_std_cat = torch.cat([mean, std], dim=1)
        attn2 = self.attn_mean_std(mean_std_cat)
        enhanced_features = features * attn1 * attn2
        return enhanced_features

class ColorCorrectionBlock(nn.Module):
    """
    A deterministic color correction block inspired by Algorithm 1 in MSDC-Net.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        x_out = []
        for img in x:
            r, g, b = img[0], img[1], img[2]
            means = {'r': torch.mean(r), 'g': torch.mean(g), 'b': torch.mean(b)}
            sorted_means = sorted(means.items(), key=lambda item: item[1])
            ch_low, mu_low = sorted_means[0]
            ch_mid, mu_mid = sorted_means[1]
            ch_high, mu_high = sorted_means[2]
            adjusted_channels = {}
            for ch_name, ch_tensor in {'r': r, 'g': g, 'b': b}.items():
                if ch_name == ch_low:
                    adjusted_channels[ch_name] = ch_tensor + (mu_mid - mu_low)
                elif ch_name == ch_high:
                    adjusted_channels[ch_name] = ch_tensor - (mu_high - mu_mid)
                else:
                    adjusted_channels[ch_name] = ch_tensor
            corrected_img = torch.stack([
                torch.clamp(adjusted_channels['r'], 0, 1),
                torch.clamp(adjusted_channels['g'], 0, 1),
                torch.clamp(adjusted_channels['b'], 0, 1)], dim=0)
            x_out.append(corrected_img)
        return torch.stack(x_out, dim=0)

class ScaleAwareFeatureAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mult_scale_heads = 8
        self.query_conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, stride=4)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)

    def forward(self, x, quarter_scale_x):
        orig_x = x
        x = self.query_conv1(x)
        x = self.query_conv2(x)
        quarter_scale_x = self.key_conv(quarter_scale_x)
        batch_size, C, roi_h, roi_w = x.size()
        x = x.view(batch_size, 1, C, roi_h, roi_w)
        quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)
        x = torch.cat((x, quarter_scale_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()
        x_embed = x
        c_embed = x_embed.size(2)
        x_embed = x_embed.view(batch_size, img_n, self.mult_scale_heads, -1, roi_h, roi_w)
        target_x_embed = x_embed[:, [1]]
        ada_weights = torch.sum(x_embed * target_x_embed, dim=3, keepdim=True) / (float(c_embed / self.mult_scale_heads)**0.5)
        ada_weights = ada_weights.expand(-1, -1, -1, int(c_embed / self.mult_scale_heads), -1, -1).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)
        x = (x * ada_weights).sum(dim=1)
        aggregated_feature = F.interpolate(x, size=orig_x.shape[-2:], mode='bilinear', align_corners=False)
        aggregated_enhanced_representation = orig_x + aggregated_feature
        return aggregated_enhanced_representation

class AquaFeat(nn.Module):
    def __init__(self, in_channels=3):
        super(AquaFeat, self).__init__()
        int_out_channels, out_channels = 32, 24
        self.relu = nn.ReLU(inplace=True)
        self.color_corrector = ColorCorrectionBlock()
        self.e_conv1 = SpecialConv(in_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)
        self.ue_conv8 = nn.Conv2d(out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.ue_conv9 = SpecialConv(int_out_channels, 3, 3, 1, 1, bias=True)
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)
        self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=out_channels)

    def forward(self, x):
        original_x = x
        corrected_x = self.color_corrector(x)
        
        quarter_scale_x = self.quarter_conv(corrected_x)
        hexa_scale_x = self.hexa_conv(quarter_scale_x)

        x1 = self.relu(self.e_conv1(corrected_x)); quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x)); hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))
        x2 = self.relu(self.e_conv2(x1)); quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1)); hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))
        x3 = self.relu(self.e_conv3(x2)); quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2)); hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))
        x4 = self.relu(self.e_conv4(x3)); quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3)); hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1))); quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1))); hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1))); quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1))); hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))
        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1))); quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1))); hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))
        x7 = self.scale_aware_aggregation(x7, quarter_scale_x7)
        hexa_scale_x7 = F.interpolate(hexa_scale_x7, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))
        enhancement_residual = torch.tanh(self.ue_conv9(x8))
        return torch.clamp(original_x + enhancement_residual, 0, 1)

print("Custom module classes, including SpecialConv, defined and modified.")

# ===================================================================
# PART 2: INJECT THE CUSTOM MODULE 
# ===================================================================
setattr(tasks, 'AquaFeat', AquaFeat)
print("AquaFeat module successfully injected into Ultralytics.")

# ===================================================================
# PART 3: CREATE THE NEW MODEL YAML 
# ===================================================================
# This yaml can be found in the official YOLO repository, you can copy and
# paste different yolo models here
yaml_content = """
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv10s model configuration file with custom FeatEnHancer
# This version has a corrected backbone and head structure to resolve runtime errors.

nc: NC_PLACEHOLDER  # Defines number of classes at the top level

# Backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, AquaFeat, [3]]             # Layer 0: Custom Enhancement. Stride 1

  # Official YOLOv10s backbone shifted by 1
  - [0, 1, Conv, [32, 3, 2]]               # Layer 1: Stride 2
  - [-1, 1, C2f, [64, True]]               # Layer 2: Stride 4
  - [-1, 1, Conv, [128, 3, 2]]             # Layer 3: Stride 8
  - [-1, 2, C2f, [256, True]]              # Layer 4: Stride 8. (P3 source)
  - [-1, 1, Conv, [512, 3, 2]]             # Layer 5: Stride 16
  - [-1, 2, C2f, [512, True]]              # Layer 6: Stride 16. (P4 source)
  - [-1, 1, Conv, [512, 3, 2]]             # Layer 7: Stride 32
  - [-1, 2, C2f, [512, True]]              # Layer 8: Stride 32
  - [-1, 1, SPPF, [512, 5]]                # Layer 9: Stride 32. (P5 source)

# Head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # Layer 10: from L9, Stride 32->16
  - [[-1, 6], 1, Concat, [1]]              # Layer 11: Concat(L10@S16, L6@S16)
  - [-1, 2, C2f, [512]]                    # Layer 12: Stride 16

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # Layer 13: from L12, Stride 16->8
  - [[-1, 4], 1, Concat, [1]]              # Layer 14: Concat(L13@S8, L4@S8)
  - [-1, 2, C2f, [256]]                    # Layer 15: Stride 8. (P3 head output)

  - [-1, 1, Conv, [256, 3, 2]]             # Layer 16: from L15, Stride 8->16
  - [[-1, 12], 1, Concat, [1]]             # Layer 17: Concat(L16@S16, L12@S16)
  - [-1, 2, C2f, [512]]                    # Layer 18: Stride 16. (P4 head output)

  - [-1, 1, Conv, [512, 3, 2]]             # Layer 19: from L18, Stride 16->32
  - [[-1, 9], 1, Concat, [1]]              # Layer 20: Concat(L19@S32, L9@S32)
  - [-1, 2, C2f, [512]]                    # Layer 21: Stride 32. (P5 head output)

  - [[15, 18, 21], 1, Detect, [NC_PLACEHOLDER]]  # Detect(P3, P4, P5)
"""

# ===================================================================
# PART 4: CUSTOM WEIGHT LOADING LOGIC 
# ===================================================================
print("\n--- Starting Custom Weight Loading ---")

pretrained_model_path = 'yolov10s.pt'

print(f"Loading pre-trained model from: {pretrained_model_path}")
try:
    og_model = YOLO(pretrained_model_path)
    og_state_dict = og_model.state_dict()
    nc = og_model.model.yaml['nc']
    print(f"Pre-trained model weights loaded. Found nc={nc} classes.")
except Exception as e:
    print(f"Could not load pre-trained model '{pretrained_model_path}'. Please ensure it's available.")
    print(f"Error: {e}")
    exit()

final_yaml_content = yaml_content.replace('NC_PLACEHOLDER', str(nc))
model_yaml_filename = "yolov10s_enhancer_transfer.yaml"
with open(model_yaml_filename, "w") as f:
    f.write(final_yaml_content)
print(f"'{model_yaml_filename}' created successfully with nc={nc}.")

print(f"Initializing new custom model from: {model_yaml_filename}")
custom_model = YOLO(model_yaml_filename)
custom_state_dict = custom_model.state_dict()
print("New custom model initialized.")

new_state_dict = OrderedDict()
print("Mapping weights from pre-trained model to new custom model...")

for k, v in og_state_dict.items():
    key_parts = k.split('.')
    if key_parts[0] == 'model' and key_parts[1].isdigit():
        original_layer_idx = int(key_parts[1])
        new_layer_idx = original_layer_idx + 1
        key_parts[1] = str(new_layer_idx)
        new_key = ".".join(key_parts)
    else:
        new_key = k

    if new_key in custom_state_dict and custom_state_dict[new_key].shape == v.shape:
        new_state_dict[new_key] = v
    else:
        print(f"  [!] Skipped {k}: Mismatch or key '{new_key}' not found in new model.")

custom_model.load_state_dict(new_state_dict, strict=False)
print("Weight transfer complete. Enhancement layer remains randomly initialized.")

# ===================================================================
# PART 5: START TRAINING (UNCHANGED)
# ===================================================================
# --- IMPORTANT: Update this path to your actual dataset.yaml file ---
data_yaml_path = "/your/path"
project_name = "Project_name"
run_name = "run_name"

print(f"\nStarting training using data from {data_yaml_path}...")

results = custom_model.train( # Our best results were obtained using this configurations
    data=data_yaml_path,
    epochs=150,
    imgsz=608,
    batch=6,
    optimizer='AdamW',
    lr0=0.0003,
    project=project_name,
    name=run_name,
    exist_ok=True,
    patience=50
)

print("Training finished.")