import os
import sys
import cv2
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)) + '/../../deps', "aanet")
sys.path.append(ROOT_DIR)

AVAILABLE = True

try:
    import torch
    import torch.nn.functional as F
    from utils import utils
    import nets
    from dataloader import transforms
except Exception as e:
    print('Failed to import aanet: ' + str(e))
    print('aanet matcher is not available')
    AVAILABLE = False

max_disp = 192  # Max disparity
feature_type = "aanet"  # Type of feature extractor
no_feature_mdconv = False  # Whether to use mdconv for feature extraction
feature_pyramid = True  # Use pyramid feature
feature_pyramid_network = True  # Use FPN
feature_similarity = "correlation"  # Similarity measure for matching cost
num_downsample = 2  # Number of downsample layer for feature extraction
aggregation_type = "adaptive"  # Type of cost aggregation
num_scales = 3  # Number of stages when using parallel aggregation
num_fusions = 6  # Number of multi-scale fusions when using parallel
num_stage_blocks = 1  # Number of deform blocks for ISA
num_deform_blocks = 3  # Number of DeformBlocks for aggregation
no_intermediate_supervision = True  # Whether to add intermediate supervision
deformable_groups = 2  # Number of deformable groups
mdconv_dilation = 2  # Dilation rate for deformable conv
refinement_type = "stereodrnet"  # Type of refinement module

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class AANetMatcher:
    def __init__(self, config):
        if AVAILABLE == False:
            return

        # Sanity check
        if not torch.cuda.is_available():
            print("GPU environment not available")
            sys.exit()

        # Setup network
        self.device = torch.device("cuda:0")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

        model_path = config['aanet']['model']

        self.model = nets.AANet(
            max_disp,
            num_downsample=num_downsample,
            feature_type=feature_type,
            no_feature_mdconv=no_feature_mdconv,
            feature_pyramid=feature_pyramid,
            feature_pyramid_network=feature_pyramid_network,
            feature_similarity=feature_similarity,
            aggregation_type=aggregation_type,
            num_scales=num_scales,
            num_fusions=num_fusions,
            num_stage_blocks=num_stage_blocks,
            num_deform_blocks=num_deform_blocks,
            no_intermediate_supervision=no_intermediate_supervision,
            refinement_type=refinement_type,
            mdconv_dilation=mdconv_dilation,
            deformable_groups=deformable_groups,
        )

        actual_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + '/../../models', model_path)

        self.model = self.model.to(self.device)
        utils.load_pretrained_net(self.model, actual_model_path, no_strict=True)

        self.model.eval()

    def process_pair(self, rectified_pair):
        if AVAILABLE == False:
            return None

        # See: https://github.com/xmba15/aanet_stereo_matching_ros/blob/master/scripts/aanet_stereo_matching.py
        left_rgb = cv2.cvtColor(rectified_pair[0], cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(rectified_pair[1], cv2.COLOR_GRAY2RGB)
        left_rectified_image = left_rgb.astype(np.float32)
        right_rectified_image = right_rgb.astype(np.float32)

        sample = {"left": left_rectified_image, "right": right_rectified_image}
        sample = self.transform(sample)
        left = sample["left"].to(self.device)  # [3, H, W]
        left = left.unsqueeze(0)  # [1, 3, H, W]
        right = sample["right"].to(self.device)
        right = right.unsqueeze(0)

        ori_height, ori_width = left.size()[2:]
        img_height = 576
        img_width = 960

        if ori_height < img_height or ori_width < img_width:
            top_pad = img_height - ori_height
            right_pad = img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            left = F.pad(left, (0, right_pad, top_pad, 0))
            right = F.pad(right, (0, right_pad, top_pad, 0))

        with torch.no_grad():
            pred_disp = self.model(left, right)[-1]  # [B, H, W]

        if pred_disp.size(-1) < left.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)), mode="bilinear") * (
                left.size(-1) / pred_disp.size(-1)
            )
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        # Crop
        if ori_height < img_height or ori_width < img_width:
            if right_pad != 0:
                pred_disp = pred_disp[:, top_pad:, :-right_pad]
            else:
                pred_disp = pred_disp[:, top_pad:]

        disp = pred_disp[0].detach().cpu().numpy()  # [H, W]

        return disp.astype(np.float32)