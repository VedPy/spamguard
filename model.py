from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent / "models"

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)
classifier = joblib.load(
    MODEL_DIR / "spam_classifier.joblib"
)

spam_index = next(
    i for i, label in enumerate(classifier.classes_)
    if str(label).lower() == "spam"
)


def predict_messages(messages):
    if not messages or any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise ValueError("Provide non-empty messages.")

    features = vectorizer.transform(messages)
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    return [
        {
            "message": message,
            "prediction": str(prediction).upper(),
            "spam_probability": float(probability[spam_index])
        }
        for message, prediction, probability in zip(
            messages, predictions, probabilities
        )
    ]