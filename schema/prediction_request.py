from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    email: str = Field(
        ...,
        description="Email Content of Customer support ticket",
    )
