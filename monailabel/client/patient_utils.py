import logging
import requests
from typing import Dict, List, Optional

from MONAILabelLib import MONAILabelClient

logger = logging.getLogger(__name__)

class PatientClient:
    """
    Extension of the MONAILabelClient with patient-specific functionality
    """
    
    def __init__(self, client: MONAILabelClient):
        self.client = client
    
    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled on the server"""
        try:
            return self.client.auth_enabled()
        except Exception as e:
            logger.error(f"Failed to check authentication status: {e}")
            return False
    
    def get_all_patients(self) -> Dict[str, List[str]]:
        """
        Get all patients and their associated images
        """
        try:
            return self.client._client.get_with_params("/patient/list", {}).json()
        except Exception as e:
            logger.error(f"Failed to get patient list: {e}")
            return {}
    
    def get_patient_images(self, patient_id: str) -> List[str]:
        """
        Get all images associated with a specific patient
        """
        try:
            return self.client._client.get_with_params(f"/patient/{patient_id}/images", {}).json()
        except Exception as e:
            logger.error(f"Failed to get images for patient {patient_id}: {e}")
            return []
