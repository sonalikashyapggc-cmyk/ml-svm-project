
import joblib
import pandas as pd

def predict(model_path, input_path):
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    preds = predict("models/svm_model.pkl", "data/processed/data.csv")
    print(preds)
