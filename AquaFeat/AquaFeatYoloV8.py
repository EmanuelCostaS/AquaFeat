"""
Official implementation for AquaFeat-YOLOv8.

This script integrates the AquaFeat enhancement module into the Ultralytics
YOLOv8 backbone for improved underwater object detection. It includes:
1.  Definitions for custom PyTorch modules:
    -   `AquaFeat`: The core enhancement module.
    -   `SpecialConv`: A trainable contrast/color enhancement layer.
    -   `ColorCorrectionBlock`: A deterministic pre-processing layer.
    -   `ScaleAwareFeatureAggregation`: A multi-scale aggregation mechanism.
2.  Logic to dynamically build a new YOLOv8 model YAML.
3.  A weight-transfer function to load pre-trained backbone weights into the
    new model architecture, leaving the custom AquaFeat layers to be trained.
4.  A main training routine configurable via command-line arguments.

Example Usage:
    python AquaFeatYoloV8.py \
        --weights /path/to/yolov8m.pt \
        --data /path/to/your_dataset.yaml \
        --project yolov8_aquafeat_runs \
        --name aquafeat_yolov8m_experiment_1 \
        --epochs 150 \
        --batch 8 \
        --imgsz 640
"""

import os
import yaml
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn import tasks
from collections import OrderedDict

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===================================================================
# PART 1: DEFINE THE CUSTOM MODULES
# ===================================================================

class SpecialConv(nn.Module):
    """
    Implementation of the Special Convolutional Layer from the MSDC-Net paper.
    
    This layer enhances contrast adjustment and color correction by using
    channel-wise statistics (mean and std) to create adaptive feature
    multipliers. It is integrated into the AquaFeat module.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        """
        Initializes the SpecialConv layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding for the convolution. Defaults to 1.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super(SpecialConv, self).__init__()
        
        # Main convolutional branch
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Adaptive branches based on channel statistics
        # Branch 1: Uses mean only
        self.attn_mean = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )
        # Branch 2: Uses concatenated mean and standard deviation
        self.attn_mean_std = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the SpecialConv layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Enhanced feature tensor.
        """
        # Main convolution
        features = self.conv(x)
        
        # --- Adaptive Statistics Calculation ---
        with torch.no_grad():
            b, c, h, w = x.shape
            mean = torch.mean(x.view(b, c, -1), dim=2).view(b, c, 1, 1)
            std = torch.std(x.view(b, c, -1), dim=2).view(b, c, 1, 1)

        # --- Adaptive Attention Generation ---
        attn1 = self.attn_mean(mean)
        
        mean_std_cat = torch.cat([mean, std], dim=1)
        attn2 = self.attn_mean_std(mean_std_cat)

        enhanced_features = features * attn1 * attn2
        
        return enhanced_features

class ColorCorrectionBlock(nn.Module):
    """
    A deterministic color correction block inspired by Algorithm 1 in MSDC-Net.
    
    This is not a trainable layer but a pre-processing step run on the GPU to
    standardize color distributions by shifting channel intensities based on
    their means.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad() # This operation does not require gradients
    def forward(self, x):
        """
        Applies deterministic color correction to a batch of images.

        Args:
            x (torch.Tensor): Batch of images, shape (B, C, H, W), normalized to [0, 1].

        Returns:
            torch.Tensor: Batch of color-corrected images.
        """
        x_out = []
        # Process each image in the batch individually
        # Note: This loop is clear but not fully vectorized. For a production
        # system, one might explore a vectorized solution if this becomes a
        # bottleneck.
        for img in x:
            # Split channels
            r, g, b = img[0], img[1], img[2]
            
            # Calculate mean of each channel
            means = {'r': torch.mean(r), 'g': torch.mean(g), 'b': torch.mean(b)}
            
            # Sort means to find the channel with the lowest, middle, and highest mean
            sorted_means = sorted(means.items(), key=lambda item: item[1])
            ch_low, mu_low = sorted_means[0]
            ch_mid, mu_mid = sorted_means[1]
            ch_high, mu_high = sorted_means[2]
            
            # Create a dictionary to hold the adjusted channels
            adjusted_channels = {}
            
            # Apply the color shift based on the paper's algorithm
            # Compensate the most absorbed color, and limit the least absorbed one.
            for ch_name, ch_tensor in {'r': r, 'g': g, 'b': b}.items():
                if ch_name == ch_low:
                    adjusted_channels[ch_name] = ch_tensor + (mu_mid - mu_low)
                elif ch_name == ch_high:
                    adjusted_channels[ch_name] = ch_tensor - (mu_high - mu_mid)
                else: # Middle channel remains unchanged as the pivot
                    adjusted_channels[ch_name] = ch_tensor
            
            # Recombine channels and clamp values to the valid [0, 1] range
            corrected_img = torch.stack([
                torch.clamp(adjusted_channels['r'], 0, 1),
                torch.clamp(adjusted_channels['g'], 0, 1),
                torch.clamp(adjusted_channels['b'], 0, 1)
            ], dim=0)
            x_out.append(corrected_img)

        # Stack the individually processed images back into a single batch tensor
        return torch.stack(x_out, dim=0)

class ScaleAwareFeatureAggregation(nn.Module):
    """
    Aggregates features from different scales using a lightweight
    attention-like mechanism.
    """
    def __init__(self, channels):
        """
        Initializes the aggregation layer.

        Args:
            channels (int): Number of input and output channels.
        """
        super().__init__()
        self.mult_scale_heads = 8
        self.query_conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, stride=4)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)

    def forward(self, x, quarter_scale_x):
        """
        Forward pass for scale-aware aggregation.

        Args:
            x (torch.Tensor): Features from the primary scale.
            quarter_scale_x (torch.Tensor): Features from the quarter scale.

        Returns:
            torch.Tensor: Aggregated and enhanced feature representation.
        """
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
    """
    The AquaFeat Enhancement Module.
    
    This module is designed to be prepended to a detection backbone (like YOLOv8)
    to perform adaptive underwater image enhancement. It combines deterministic
    color correction with a trainable, multi-scale U-Net-like architecture
    that uses SpecialConv layers for adaptive feature transformation.
    """
    def __init__(self, in_channels=3):
        """
        Initializes the AquaFeat module.

        Args:
            in_channels (int, optional): Number of input channels (e.g., 3 for RGB).
                                       Defaults to 3.
        """
        super(AquaFeat, self).__init__()
        int_out_channels = 32
        out_channels = 24  # Used for intermediate features
        self.relu = nn.ReLU(inplace=True)
        
        # Instantiate the deterministic Color Correction Block
        self.color_corrector = ColorCorrectionBlock()
        
        # Use SpecialConv for the first layer to adapt to the input
        self.e_conv1 = SpecialConv(in_channels, int_out_channels, 3, 1, 1, bias=True)
        
        # Standard convolutions for intermediate layers
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)
        self.ue_conv8 = nn.Conv2d(out_channels * 2, int_out_channels, 3, 1, 1, bias=True)

        # Use SpecialConv for the final layer to produce an adaptive residual
        self.ue_conv9 = SpecialConv(int_out_channels, 3, 3, 1, 1, bias=True)

        # Downsampling convolutions for multi-scale processing
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)
        
        # Scale aggregation module
        self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=out_channels)

    def forward(self, x):
        """
        Forward pass of the AquaFeat module.

        Args:
            x (torch.Tensor): Input image batch, normalized to [0, 1].

        Returns:
            torch.Tensor: Enhanced image batch, clamped to [0, 1].
        """
        original_x = x # Keep a reference to the original input

        # Apply deterministic color correction as the very first step
        corrected_x = self.color_corrector(x)

        # --- Multi-Scale Feature Extraction ---
        quarter_scale_x = self.quarter_conv(corrected_x)
        hexa_scale_x = self.hexa_conv(quarter_scale_x)
        
        x1 = self.relu(self.e_conv1(corrected_x))
        quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
        hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

        x2 = self.relu(self.e_conv2(x1))
        quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
        hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

        x3 = self.relu(self.e_conv3(x2))
        quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
        hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))
        
        x4 = self.relu(self.e_conv4(x3))
        quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
        hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
        hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
        hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))
        
        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
        quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
        hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))
        
        # --- Aggregation and Upsampling ---
        x7 = self.scale_aware_aggregation(x7, quarter_scale_x7)
        
        hexa_scale_x7 = F.interpolate(hexa_scale_x7, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))

        # --- Final Residual ---
        enhancement_residual = torch.tanh(self.ue_conv9(x8))

        # Add residual to the original image (not the corrected one)
        # This allows the network to learn enhancement, not just reconstruction
        return torch.clamp(original_x + enhancement_residual, 0, 1)

# ===================================================================
# PART 2: MODEL DEFINITION (YAML)
# ===================================================================

# This YAML string defines the new model architecture, inserting AquaFeat
# as the very first layer (layer 0).
MODEL_YAML_TEMPLATE = """
nc: {nc}  # Number of classes (will be replaced dynamically)

# Backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, AquaFeat, [3]]          # Layer 0: Custom input enhancement.

  # The rest of the backbone now connects to Layer 0
  - [0, 1, Conv, [64, 3, 2]]         # Layer 1
  - [-1, 1, C2f, [128, True]]        # Layer 2
  - [-1, 1, Conv, [128, 3, 2]]       # Layer 3
  - [-1, 4, C2f, [256, True]]        # Layer 4
  - [-1, 1, Conv, [256, 3, 2]]       # Layer 5
  - [-1, 4, C2f, [512, True]]        # Layer 6
  - [-1, 1, Conv, [512, 3, 2]]       # Layer 7
  - [-1, 2, C2f, [512, True]]        # Layer 8
  - [-1, 1, SPPF, [512, 5]]          # Layer 9

# Head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # Layer 10
  - [[-1, 6], 1, Concat, [1]]      # Concat with layer 6
  - [-1, 2, C2f, [512]]              # Layer 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # Layer 13
  - [[-1, 4], 1, Concat, [1]]      # Concat with layer 4
  - [-1, 2, C2f, [256]]              # Layer 15

  - [-1, 1, Conv, [256, 3, 2]]       # Layer 16
  - [[-1, 12], 1, Concat, [1]]     # Concat with layer 12
  - [-1, 2, C2f, [512]]              # Layer 18

  - [-1, 1, Conv, [512, 3, 2]]       # Layer 19
  - [[-1, 9], 1, Concat, [1]]      # Concat with layer 9
  - [-1, 2, C2f, [512]]              # Layer 21

  - [[15, 18, 21], 1, Detect, [{nc}]]  # Detect(P3, P4, P5)
"""

# ===================================================================
# PART 3: MAIN SCRIPT LOGIC
# ===================================================================

def main(args):
    """
    Main function to orchestrate the model creation, weight loading,
    and training process.
    """
    
    # --- 1. Inject Custom Module ---
    # This makes 'AquaFeat' recognizable by the YOLO model parser
    setattr(tasks, 'AquaFeat', AquaFeat)
    logging.info("AquaFeat module successfully injected into Ultralytics tasks.")

    # --- 2. Load Original Model and Get NC ---
    logging.info(f"Loading original model from: {args.weights}")
    try:
        og_model = YOLO(args.weights)
        og_state_dict = og_model.state_dict()
        nc = int(og_model.model.yaml['nc'])
        logging.info(f"Original model weights loaded. Found nc={nc} classes.")
    except Exception as e:
        logging.error(f"Could not load original model or find 'nc': {e}")
        logging.error("Please ensure '--weights' points to a valid YOLOv8 .pt file.")
        return

    # --- 3. Create Custom Model YAML ---
    final_yaml_content = MODEL_YAML_TEMPLATE.format(nc=nc)
    model_yaml_filename = f"yolov8_{args.name}_temp.yaml"
    
    try:
        with open(model_yaml_filename, "w") as f:
            f.write(final_yaml_content)
        logging.info(f"'{model_yaml_filename}' created successfully with nc={nc}.")
    except IOError as e:
        logging.error(f"Failed to write YAML file: {e}")
        return

    # --- 4. Initialize Custom Model ---
    logging.info(f"Initializing new custom model from: {model_yaml_filename}")
    custom_model = YOLO(model_yaml_filename)
    custom_state_dict = custom_model.state_dict()
    logging.info("New custom model initialized.")

    # --- 5. Perform Weight Transfer ---
    new_state_dict = OrderedDict()
    logging.info("Mapping weights from original model to new custom model...")

    for k, v in og_state_dict.items():
        key_parts = k.split('.')
        
        # This logic shifts layer indices for the backbone and neck
        # model.0. ... -> model.1. ...
        # model.1. ... -> model.2. ...
        # ...
        # model.21. ... -> model.22. ... (Detect head)
        if key_parts[0] == 'model' and key_parts[1].isdigit():
            original_layer_idx = int(key_parts[1])
            # The 'Detect' layer index (21 in v8m) and beyond are not shifted
            if original_layer_idx < 21: 
                new_layer_idx = original_layer_idx + 1 # Shift backbone/neck layers
                key_parts[1] = str(new_layer_idx)
                new_key = ".".join(key_parts)
            else:
                new_key = k # Keep Detect head keys the same
        else:
            new_key = k # Keep other keys (e.g., 'updates')

        if new_key in custom_state_dict:
            if custom_state_dict[new_key].shape == v.shape:
                new_state_dict[new_key] = v
            else:
                logging.warning(
                    f"  [!] Skipped {k}: Shape mismatch. "
                    f"OG_shape={v.shape}, Custom_shape={custom_state_dict[new_key].shape}"
                )
        else:
            # We expect to skip the original layer 0 (Conv) as it's been replaced
            if k != 'model.0.conv.weight':
                logging.warning(f"  [!] Skipped {k}: Key '{new_key}' not found in new model.")

    # Load the carefully mapped weights
    custom_model.load_state_dict(new_state_dict, strict=False)
    logging.info("Weight transfer complete. AquaFeat layer (model.0) remains "
                 "randomly initialized as intended.")

    # --- 6. Start Training ---
    logging.info(f"\nStarting training using data from {args.data}...")
    
    try:
        custom_model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            lr0=args.lr0,
            project=args.project,
            name=args.name,
            exist_ok=True,
            patience=args.patience
        )
        logging.info("Training finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
    finally:
        # Clean up the temporary YAML file
        if os.path.exists(model_yaml_filename):
            os.remove(model_yaml_filename)
            logging.info(f"Cleaned up temporary file: {model_yaml_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 with AquaFeat Enhancement Module"
    )
    
    # --- Model & Data Arguments ---
    parser.add_argument(
        '--weights', 
        type=str, 
        required=True, 
        help="Path to the pre-trained YOLOv8 .pt file (e.g., yolov8m.pt)"
    )
    parser.add_argument(
        '--data', 
        type=str, 
        required=True, 
        help="Path to the dataset's .yaml file"
    )
    
    # --- Training Arguments ---
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=150, 
        help="Total number of training epochs"
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=8, 
        help="Batch size for training"
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640, 
        help="Image size for training (e.g., 640)"
    )
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default='AdamW', 
        help="Optimizer to use (e.g., 'AdamW', 'SGD')"
    )
    parser.add_argument(
        '--lr0', 
        type=float, 
        default=0.0003, 
        help="Initial learning rate"
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=50, 
        help="Epochs to wait for no improvement before early stopping"
    )
    
    # --- Output Arguments ---
    parser.add_argument(
        '--project', 
        type=str, 
        default='runs/train_aquafeat', 
        help="Project directory to save results"
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default='exp', 
        help="Name for the specific training run"
    )
    
    args = parser.parse_args()
    main(args)
