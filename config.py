from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    MISTRAL_API_KEY: str | None = None
    ACCESS_TOKEN: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    CODESTRAL_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None

    @property
    def allowed_origins_list(self) -> List[str]:
        return self.ALLOWED_ORIGINS.split(",")

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
