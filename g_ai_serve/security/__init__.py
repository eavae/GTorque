import os
from typing import Annotated
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel

from .constants import oauth2schema


class User(BaseModel):
    username: str
    disabled: bool | None = None


async def get_current_user(token: Annotated[str, Depends(oauth2schema)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(
            token, os.environ.get("JWT_SECRET_KEY"), algorithms=["HS256"]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        return User(username=username)
    except JWTError:
        raise credentials_exception


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="disabled user")
    return current_user
