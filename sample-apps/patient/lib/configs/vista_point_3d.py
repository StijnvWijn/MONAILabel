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

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool
from lib.model.vista_point_3d.vista3d import vista_model_registry

logger = logging.getLogger(__name__)


class VISTAPOINT3D(TaskConfig):
    def __init__(self):
        super().__init__()

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "background": 0,
            "liver": 1,
            "kidney": 2,
            "spleen": 3,
            "pancreas": 4,
            "right kidney": 5,
            "aorta": 6,
            "inferior vena cava": 7,
            "right adrenal gland": 8,
            "left adrenal gland": 9,
            "gallbladder": 10,
            "esophagus": 11,
            "stomach": 12,
            "duodenum": 13,
            "left kidney": 14,
            "bladder": 15,
            "prostate or uterus": 16,
            "portal vein and splenic vein": 17,
            "rectum": 18,
            "small bowel": 19,
            "lung": 20,
            "bone": 21,
            "brain": 22,
            "lung tumor": 23,
            "pancreatic tumor": 24,
            "hepatic vessel": 25,
            "hepatic tumor": 26,
            "colon cancer primaries": 27,
            "left lung upper lobe": 28,
            "left lung lower lobe": 29,
            "right lung upper lobe": 30,
            "right lung middle lobe": 31,
            "right lung lower lobe": 32,
            "vertebrae L5": 33,
            "vertebrae L4": 34,
            "vertebrae L3": 35,
            "vertebrae L2": 36,
            "vertebrae L1": 37,
            "vertebrae T12": 38,
            "vertebrae T11": 39,
            "vertebrae T10": 40,
            "vertebrae T9": 41,
            "vertebrae T8": 42,
            "vertebrae T7": 43,
            "vertebrae T6": 44,
            "vertebrae T5": 45,
            "vertebrae T4": 46,
            "vertebrae T3": 47,
            "vertebrae T2": 48,
            "vertebrae T1": 49,
            "vertebrae C7": 50,
            "vertebrae C6": 51,
            "vertebrae C5": 52,
            "vertebrae C4": 53,
            "vertebrae C3": 54,
            "vertebrae C2": 55,
            "vertebrae C1": 56,
            "trachea": 57,
            "left iliac artery": 58,
            "right iliac artery": 59,
            "left iliac vena": 60,
            "right iliac vena": 61,
            "colon": 62,
            "left rib 1": 63,
            "left rib 2": 64,
            "left rib 3": 65,
            "left rib 4": 66,
            "left rib 5": 67,
            "left rib 6": 68,
            "left rib 7": 69,
            "left rib 8": 70,
            "left rib 9": 71,
            "left rib 10": 72,
            "left rib 11": 73,
            "left rib 12": 74,
            "right rib 1": 75,
            "right rib 2": 76,
            "right rib 3": 77,
            "right rib 4": 78,
            "right rib 5": 79,
            "right rib 6": 80,
            "right rib 7": 81,
            "right rib 8": 82,
            "right rib 9": 83,
            "right rib 10": 84,
            "right rib 11": 85,
            "right rib 12": 86,
            "left humerus": 87,
            "right humerus": 88,
            "left scapula": 89,
            "right scapula": 90,
            "left clavicula": 91,
            "right clavicula": 92,
            "left femur": 93,
            "right femur": 94,
            "left hip": 95,
            "right hip": 96,
            "sacrum": 97,
            "left gluteus maximus": 98,
            "right gluteus maximus": 99,
            "left gluteus medius": 100,
            "right gluteus medius": 101,
            "left gluteus minimus": 102,
            "right gluteus minimus": 103,
            "left autochthon": 104,
            "right autochthon": 105,
            "left iliopsoas": 106,
            "right iliopsoas": 107,
            "left atrial appendage": 108,
            "brachiocephalic trunk": 109,
            "left brachiocephalic vein": 110,
            "right brachiocephalic vein": 111,
            "left common carotid artery": 112,
            "right common carotid artery": 113,
            "costal cartilages": 114,
            "heart": 115,
            "left kidney cyst": 116,
            "right kidney cyst": 117,
            "prostate": 118,
            "pulmonary vein": 119,
            "skull": 120,
            "spinal cord": 121,
            "sternum": 122,
            "left subclavian artery": 123,
            "right subclavian artery": 124,
            "superior vena cava": 125,
            "thyroid gland": 126,
            "vertebrae S1": 127,
            "bone lesion": 128,
            "kidney mass": 129,
            "liver tumor": 130,
            "vertebrae L6": 131,
            "airway": 132
        }

        self.conf['pretrained_path'] = 'https://huggingface.co/MONAI/vista3d/resolve/0.5.8/models'

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/model.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.5, 1.5, 1.5)  # target space for image
        # Setting ROI size - This is for the image padding
        self.roi_size = (128, 128, 128)

        self.network = vista_model_registry["vista3d132"](in_channels=1, image_size=self.roi_size)

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.VISTAPOINT3D(
            path=self.path,
            network=self.network,
            target_spacing=self.target_spacing,
            roi_size=self.roi_size,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        None
