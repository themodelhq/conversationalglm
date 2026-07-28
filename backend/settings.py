from __future__ import annotations
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",env_prefix="GLM_",extra="ignore")
    env:str="development"; database_url:str="sqlite+aiosqlite:///./conversational_glm.db"; jwt_secret:str="development-secret-change-before-production"; jwt_algorithm:str="HS256"; access_token_minutes:int=1440; cors_origins:str="http://localhost:5173"; cors_origin_regex:str=""; model_path:str=""; device:str="auto"; api_rate_limit:int=120; max_upload_mb:int=50; storage_dir:str=".glm-state"
    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls,value:str)->str:
        if value.startswith("postgres://"):return "postgresql+asyncpg://"+value.removeprefix("postgres://")
        if value.startswith("postgresql://"):return "postgresql+asyncpg://"+value.removeprefix("postgresql://")
        return value
    @property
    def origins(self)->list[str]:return [item.strip() for item in self.cors_origins.split(",") if item.strip()]
settings=Settings()
