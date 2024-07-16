import argparse
import jwt
import os
from passlib.context import CryptContext
from datetime import datetime, timedelta


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, os.environ.get("JWT_SECRET_KEY"), algorithm="HS256"
    )
    return encoded_jwt


if __name__ == "__main__":
    context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    args = parser.parse_args()

    token = create_access_token(
        data={"sub": args.username}, expires_delta=timedelta(days=365)
    )

    print(f"Your token generated for {args.username}")
    print(token)
