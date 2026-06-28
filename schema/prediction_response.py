from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_team: str = Field(
        ...,
        description="The predicted Team to handle the support ticket",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence score for the predicted Team (range: 0 to 1)",
    )
    urgency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Urgency Score for the ticket",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_team": "Billing",

                    "confidence": 0.8432,
                    
                    "urgency_score": 0.9354,
                }
            ]
        }
    }
