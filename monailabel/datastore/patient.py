import os
import logging
from typing import Dict, List
from monailabel.datastore.local import LocalDatastore

logger = logging.getLogger(__name__)

class PatientDatastore(LocalDatastore):
    def __init__(
        self,
        datastore_path: str,
        extensions=None,
        auto_reload=True,
        read_only=False,
    ):
        super().__init__(datastore_path, extensions, auto_reload, read_only)
        self.patient_images: Dict[str, List[str]] = {}
        self._init_patient_mapping()

    def _init_patient_mapping(self):
        # Group images by patient ID (assuming filenames start with patient ID)
        for image_id in self.image_ids():
            patient_id = image_id.split("_")[0]  # Assumes filename format: PATIENTID_*.nii.gz
            if patient_id not in self.patient_images:
                self.patient_images[patient_id] = []
            self.patient_images[patient_id].append(image_id)

    def get_patient_images(self, patient_id: str) -> List[str]:
        return self.patient_images.get(patient_id, [])

    def get_all_patients(self) -> Dict[str, List[str]]:
        return self.patient_images

    def add_image(self, image_file: str, image_id: str = None) -> str:
        image_id = super().add_image(image_file, image_id)
        patient_id = image_id.split("_")[0]
        if patient_id not in self.patient_images:
            self.patient_images[patient_id] = []
        self.patient_images[patient_id].append(image_id) 
        return image_id
