from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    search_service: str = Field(default="", validation_alias=AliasChoices("search_service", "AZURE_SEARCH_SERVICE"))
    search_index: str = Field(default="kb-docx-index", validation_alias=AliasChoices("search_index", "AZURE_SEARCH_INDEX"))
    search_key: str = Field(default="", validation_alias=AliasChoices("search_key", "AZURE_SEARCH_KEY"))

    azure_openai_service: str | None = Field(
        default=None,
        validation_alias=AliasChoices("azure_openai_service", "AZURE_OPENAI_SERVICE"),
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("azure_openai_endpoint", "AZURE_OPENAI_ENDPOINT"),
    )
    azure_openai_key: str = Field(default="", validation_alias=AliasChoices("azure_openai_key", "AZURE_OPENAI_KEY"))
    azure_openai_api_version: str = Field(
        default="2024-06-01",
        validation_alias=AliasChoices("azure_openai_api_version", "AZURE_OPENAI_API_VERSION"),
    )
    azure_openai_embedding_deployment: str = Field(
        default="",
        validation_alias=AliasChoices("azure_openai_embedding_deployment", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    azure_openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("azure_openai_embedding_model", "AZURE_OPENAI_EMBEDDING_MODEL"),
    )
    azure_openai_embedding_dimensions: int = Field(
        default=1536,
        validation_alias=AliasChoices("azure_openai_embedding_dimensions", "AZURE_OPENAI_EMBEDDING_DIMENSIONS"),
    )
    azure_openai_chat_deployment: str = Field(
        default="",
        validation_alias=AliasChoices("azure_openai_chat_deployment", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    )
    azure_openai_chat_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("azure_openai_chat_model", "AZURE_OPENAI_CHAT_MODEL"),
    )

    search_embedding_field: str = Field(
        default="embedding",
        validation_alias=AliasChoices("search_embedding_field", "SEARCH_EMBEDDING_FIELD"),
    )
    category: str = Field(default="default", validation_alias=AliasChoices("category", "CATEGORY"))

    # DOCX image preprocessing (optional)
    enable_docx_image_preprocessing: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_docx_image_preprocessing", "ENABLE_DOCX_IMAGE_PREPROCESSING"),
    )
    azure_storage_connection_string: str = Field(
        default="",
        validation_alias=AliasChoices("azure_storage_connection_string", "AZURE_STORAGE_CONNECTION_STRING"),
    )
    azure_storage_images_container_name: str = Field(
        default="document-images",
        validation_alias=AliasChoices("azure_storage_images_container_name", "AZURE_STORAGE_IMAGES_CONTAINER_NAME"),
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> "Settings":
        required_fields = (
            "search_service",
            "search_key",
            "azure_openai_key",
            "azure_openai_embedding_deployment",
            "azure_openai_chat_deployment",
        )
        missing = [field for field in required_fields if not getattr(self, field).strip()]
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")
        if self.enable_docx_image_preprocessing and not self.azure_storage_connection_string.strip():
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is required when ENABLE_DOCX_IMAGE_PREPROCESSING is true"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
