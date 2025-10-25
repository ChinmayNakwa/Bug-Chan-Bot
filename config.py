from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    ASI_ONE_MINI_API_KEY: str = Field(..., env="ASI_ONE_MINI_API_KEY")
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")


    class Config:
        extra = "allow"
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()