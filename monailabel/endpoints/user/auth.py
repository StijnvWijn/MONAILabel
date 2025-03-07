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
from typing import List, Sequence, Union
from datetime import datetime, timezone, timedelta

import jwt
import requests
import json
import os
from cachetools import cached
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel

from monailabel.config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Local users cache
_local_users: dict = None
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"

class Token(BaseModel):
    access_token: str
    token_type: str

def get_local_users() -> dict:
    """Get users from config.json file"""
    global _local_users
    
    if _local_users is not None:
        return _local_users
    
    config_file = os.path.join(settings.MONAI_LABEL_APP_DIR, "config.json")
    if not os.path.exists(config_file):
        logger.error(f"Config file not found at {config_file}, no local users available")
        _local_users = {}
        return _local_users
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            auth_config = config.get("auth", {})
            _local_users = auth_config.get("users", {})
            return _local_users
    except Exception as e:
        logger.error(f"Error loading local users from config.json: {e}")
        _local_users = {}
        return _local_users


def validate_local_user(username: str, password: str) -> dict:
    """Validate user against local users in config.json"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate local credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    users = get_local_users()
    user_info = users.get(username, None)
    
    if not user_info:
        raise credentials_exception
    
    if user_info.get("password") == password:
        return user_info
    
    raise credentials_exception

def create_local_token(username: str, user_info: dict) -> Token:
    """Create a JWT token for a local user"""
    
    payload = {
        "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.MONAI_LABEL_SESSION_EXPIRY),
        "iat": datetime.now(timezone.utc),
        "sub": username,
        settings.MONAI_LABEL_AUTH_TOKEN_USERNAME: username,
        settings.MONAI_LABEL_AUTH_TOKEN_EMAIL: user_info.get("email", ""),
        settings.MONAI_LABEL_AUTH_TOKEN_NAME: username,
        "local_auth": True
    }
    
    # Add roles based on config format
    roles_key = settings.MONAI_LABEL_AUTH_TOKEN_ROLES.split("#")[0]
    payload[roles_key] = user_info.get("roles", [])
    
    token = jwt.encode(payload, SECRET_KEY)
    return Token(access_token=token, token_type="bearer")

@cached(cache={})
def get_public_key(realm_uri) -> str:
    """Get the public key from the realm URI"""
    logger.info(f"Fetching public key for: {realm_uri}")
    r = requests.get(url=realm_uri, timeout=settings.MONAI_LABEL_AUTH_TIMEOUT)
    r.raise_for_status()
    j = r.json()

    key = j["public_key"]
    return f"-----BEGIN PUBLIC KEY-----\n{key}\n-----END PUBLIC KEY-----"


@cached(cache={})
def open_id_configuration(realm_uri):
    """Get the OpenID configuration from the realm URI"""
    response = requests.get(
        url=f"{realm_uri}/.well-known/openid-configuration",
        timeout=settings.MONAI_LABEL_AUTH_TIMEOUT,
    )
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Error {e} occurred when loading retrieving token from {realm_uri}/.well-known/openid-configuration", exc_info=True)
        logger.error(f"Response: {response}, have you set the correct realm URI?")
        return {}


def token_uri():
    return open_id_configuration(settings.MONAI_LABEL_AUTH_REALM_URI).get("token_endpoint")


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    name: Union[str, None] = None
    roles: List[str] = []


DEFAULT_USER = User(
    username="admin",
    email="admin@monailabel.com",
    name="UNK",
    roles=[
        settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
        settings.MONAI_LABEL_AUTH_ROLE_REVIEWER,
        settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
        settings.MONAI_LABEL_AUTH_ROLE_USER,
    ],
)


def from_token(token: str):
    """Decode a JWT token and return a User object"""
    if not settings.MONAI_LABEL_AUTH_ENABLE:
        return DEFAULT_USER

    options = {
        "verify_signature": True,
        "verify_aud": False,
        "verify_exp": True,
    }
    # If realm URI is set, use the public key from the realm, otherwise use the default secret key
    if settings.MONAI_LABEL_AUTH_REALM_URI:
        key = get_public_key(settings.MONAI_LABEL_AUTH_REALM_URI)
    else:
        key = SECRET_KEY
    payload = jwt.decode(token, key, options=options)

    username: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_USERNAME)
    email: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_EMAIL)
    name: str = payload.get(settings.MONAI_LABEL_AUTH_TOKEN_NAME)

    kr = settings.MONAI_LABEL_AUTH_TOKEN_ROLES.split("#")
    if len(kr) > 1:
        p = payload
        for r in kr:
            roles = p.get(r)
            p = roles
    else:
        roles = payload.get(kr[0])
    roles = [] if not roles else roles

    return User(username=username, email=email, name=name, roles=roles)


async def get_current_user(token: str = Depends(oauth2_scheme) if settings.MONAI_LABEL_AUTH_ENABLE else ""):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        return from_token(token)
    except InvalidTokenError as e:
        logger.error(e)
        raise credentials_exception


class RBAC:
    def __init__(self, roles: Union[str, Sequence[str]]):
        self.roles = roles

    async def __call__(self, user: User = Security(get_current_user)):
        if not settings.MONAI_LABEL_AUTH_ENABLE:
            return user

        roles = self.roles
        if isinstance(roles, str):
            roles = (
                [roles]
                if roles != "*"
                else [
                    settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
                    settings.MONAI_LABEL_AUTH_ROLE_REVIEWER,
                    settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
                    settings.MONAI_LABEL_AUTH_ROLE_USER,
                ]
            )

        for role in roles:
            if role in user.roles:
                return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Role "{role}" is required to perform this action',
        )
