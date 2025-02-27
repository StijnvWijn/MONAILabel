
import logging
from fastapi import APIRouter, Depends
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/patient",
    tags=["Patient"],
    responses={404: {"description": "Not found"}},
)

@router.get("/list", summary="Get list of all patients")
async def api_patient_list(user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))):
    instance: MONAILabelApp = app_instance()
    if hasattr(instance.datastore(), "get_all_patients"):
        return instance.datastore().get_all_patients()
    return {}

@router.get("/{patient_id}/images", summary="Get all images for a patient")
async def api_patient_images(
    patient_id: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))
):
    instance: MONAILabelApp = app_instance()
    if hasattr(instance.datastore(), "get_patient_images"):
        return instance.datastore().get_patient_images(patient_id)
    return []
