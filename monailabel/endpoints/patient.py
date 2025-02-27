import logging
from fastapi import APIRouter, Depends
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.config import RBAC_USER, settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/patient",
    tags=["Patient"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"}
    },
)

@router.get("/list", summary=f"{RBAC_USER}Get list of all patients")
async def api_patient_list(user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))):
    """
    Get list of all patients with their associated images.
    This endpoint requires user authentication.
    """
    instance: MONAILabelApp = app_instance()
    if hasattr(instance.datastore(), "get_all_patients"):
        return instance.datastore().get_all_patients()
    return {}

@router.get("/{patient_id}/images", summary=f"{RBAC_USER}Get all images for a patient")
async def api_patient_images(
    patient_id: str,
    user: User = Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))
):
    """
    Get all images associated with a specific patient.
    This endpoint requires user authentication.
    """
    instance: MONAILabelApp = app_instance()
    if hasattr(instance.datastore(), "get_patient_images"):
        return instance.datastore().get_patient_images(patient_id)
    return []
