from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_ENV: str = "development"

    OPENAI_API_KEY: str

    VECTOR_DB_TYPE: str = "faiss"
    VECTOR_DB_PATH: str = "./vectorstore/vector_index.faiss"

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
