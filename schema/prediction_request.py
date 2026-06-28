from pydantic import BaseModel


class PredictionRequest(BaseModel):
    email: str
