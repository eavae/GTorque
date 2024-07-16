from fastapi.security import OAuth2PasswordBearer


oauth2schema = OAuth2PasswordBearer(tokenUrl="token")
