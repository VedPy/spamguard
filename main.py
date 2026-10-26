from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import predict_messages
from knapsack import knapsack_01

app = FastAPI(title="SMS Spam Classifier")

BASE_DIR = Path(__file__).resolve().parent


class ReviewRequest(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=100)
    review_times: list[int]
    time_budget: int = Field(ge=0, le=600)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize")
def optimize(request: ReviewRequest):
    messages = [message.strip() for message in request.messages]

    if any(not message or len(message) > 10000 for message in messages):
        raise HTTPException(
            status_code=400,
            detail="Messages must contain 1 to 10000 characters."
        )

    if len(messages) != len(request.review_times):
        raise HTTPException(
            status_code=400,
            detail="Provide one review time per message."
        )

    if any(
        not isinstance(t, int) or isinstance(t, bool) or t < 1 or t > 60
        for t in request.review_times
    ):
        raise HTTPException(
            status_code=400,
            detail="Review times must be between 1 and 60 minutes."
        )

    predictions = predict_messages(messages)

    # Only messages predicted as SPAM are eligible for review.
    candidates = [
        i for i, result in enumerate(predictions)
        if result["prediction"] == "SPAM"
    ]

    values = [
        round(predictions[i]["spam_probability"] * 100)
        for i in candidates
    ]

    weights = [request.review_times[i] for i in candidates]

    max_value, selected = knapsack_01(
        values, weights, request.time_budget
    )

    selected_indices = {candidates[i] for i in selected}
    results = []

    for i, result in enumerate(predictions):
        is_spam = result["prediction"] == "SPAM"

        results.append({
            **result,
            "review_value": (
                round(result["spam_probability"] * 100)
                if is_spam else 0
            ),
            "review_time": request.review_times[i],
            "decision": (
                "EXCLUDED (HAM)" if not is_spam
                else "SELECTED" if i in selected_indices
                else "NOT SELECTED"
            )
        })

    return {
        "results": results,
        "summary": {
            "maximum_value": max_value,
            "time_used": sum(
                request.review_times[i] for i in selected_indices
            ),
            "time_budget": request.time_budget,
            "selected_count": len(selected_indices)
        }
    }
