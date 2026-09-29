from pydantic import BaseModel


class ReadinessResponse(BaseModel):
    status: str
    postgres: str
    qdrant: str
