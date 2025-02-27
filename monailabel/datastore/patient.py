import os
import logging
from typing import Dict, List, Optional, Union
from monailabel.datastore.local import LocalDatastore

logger = logging.getLogger(__name__)

class PatientDatastore(LocalDatastore):
    def __init__(
        self,
        datastore_path: str,
        extensions: Optional[List[str]] = None,
        auto_reload: bool = True,
        read_only: bool = False,
    ):
        # Pass all parameters explicitly to avoid issues with default arguments
        super().__init__(
            datastore_path=datastore_path,
            extensions=extensions if extensions else None,
            auto_reload=auto_reload,
            read_only=read_only
        )
        self.patient_images: Dict[str, List[str]] = {}
        self._init_patient_mapping()

    def _init_patient_mapping(self):
        # Group images by patient ID (assuming filenames start with patient ID)
        # Use list_images() instead of image_ids() - this is the method defined in LocalDatastore
        for image_id in self.list_images():
            patient_id = image_id.split("_")[0]  # Assumes filename format: PATIENTID_*.nii.gz
            if patient_id not in self.patient_images:
                self.patient_images[patient_id] = []
            self.patient_images[patient_id].append(image_id)
        logger.info(f"Found {len(self.patient_images)} patients with {len(self.list_images())} total images")

    def get_patient_images(self, patient_id: str) -> List[str]:
        return self.patient_images.get(patient_id, [])

    def get_all_patients(self) -> Dict[str, List[str]]:
        return self.patient_images

    def add_image(self, image_file: str, image_id: str = None) -> str:
        image_id = super().add_image(image_file, image_id)
        patient_id = image_id.split("_")[0]
        if patient_id not in self.patient_images:
            self.patient_images[patient_id] = []
        if image_id not in self.patient_images[patient_id]:
            self.patient_images[patient_id].append(image_id)
        return image_id
        
    def remove_image(self, image_id: str) -> None:
        # Override to maintain patient mapping when images are removed
        patient_id = image_id.split("_")[0]
        if patient_id in self.patient_images and image_id in self.patient_images[patient_id]:
            self.patient_images[patient_id].remove(image_id)
            # Remove patient entry if no more images
            if not self.patient_images[patient_id]:
                del self.patient_images[patient_id]
                
        # Call parent implementation to remove the image
        super().remove_image(image_id)
        
    def refresh(self):
        # Override refresh to update patient mapping
        super().refresh()
        # Rebuild patient mapping
        self.patient_images.clear()
        self._init_patient_mapping()
