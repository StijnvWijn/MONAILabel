# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence

from monailabel.tasks.infer.basic_infer import BasicInferTask
from lib.model.vista_point_3d.inferer import VISTAPOINT3DInferer
from monai.inferers import Inferer
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    CastToTyped,
    Invertd,
    Activationsd,
    AsDiscreted,
    AsDiscrete,
    Compose,
    Resize,
)
from monai.data import decollate_batch
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored
import torch
from lib.transforms.transforms import ThreshMergeLabeld
from typing import Dict, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class VISTAPOINT3D(BasicInferTask):
    """
    This provides Inference Engine for pre-trained VISTA segmentation model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.5, 1.5, 1.5),
        roi_size=(96, 96, 96),
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric segmentation using VISTA3D",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.target_spacing = target_spacing
        self.roi_size = roi_size

    def is_valid(self) -> bool:
        return True

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=-963.8247715525971, a_max=1053.678477684517, b_min=0.0, b_max=1.0, clip=True),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=self.target_spacing, mode="bilinear", align_corners=True),
            CastToTyped(keys="image", dtype=torch.float32),
        ]

    def inferer(self, data=None) -> Inferer:
        return VISTAPOINT3DInferer(device=data.get("device") if data else None, roi_size=self.roi_size)
    
    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")

        network = self._get_network(device, data)
        
        inputs = data[self.input_key]
        # point prompts are given separately in the data from the UI
        foreground_points = data.get("foreground", [])
        background_points = data.get("background", [])
        # VISTA3D expects points in the format [bs, N, 3], with the foreground and background points concatenated
        point_prompts = foreground_points + background_points
        # We label foreground points as 1 and background points as 0, as required by VISTA3D
        point_labels = [1] * len(foreground_points) + [0] * len(background_points)
        inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
        inputs = inputs[None] if convert_to_batch else inputs
        inputs = inputs.to(torch.device(device))
        # label prompt is given as a str that corresponds to a label in the labels dict
        label_prompt = data.get("label", None)
        class_prompt = self.labels.get(label_prompt, -1) if label_prompt is not None else None
        class_prompt = [class_prompt]

        with torch.no_grad():
            outputs = inferer(inputs, network, point_prompts=point_prompts, point_labels=point_labels, class_prompts=class_prompt)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        if convert_to_batch:
            if isinstance(outputs, dict):
                outputs_d = decollate_batch(outputs)
                outputs = outputs_d[0]
            else:
                outputs = outputs[0]

        data[self.output_label_key] = outputs
        return data

    def inverse_transforms(self, data=None):
        return self.pre_transforms(data)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            # Invertd(	
            #     keys="pred",	
            #     transform=self.infer_transforms,	
            #     orig_keys="image",	
            #     meta_keys="pred_meta_dict",	
            #     orig_meta_keys="image_meta_dict",	
            #     meta_key_postfix="meta_dict",	
            #     nearest_interp=False,	
            #     to_tensor=True
            # ),	
            # Activationsd(	
            #     keys="pred",	
            #     softmax=False,	
            #     sigmoid=True
            # ),
            # AsDiscreted(
            #     keys="pred",
            #     threshold=0.5
            # ),
            ThreshMergeLabeld(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]