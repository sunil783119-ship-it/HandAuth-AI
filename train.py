import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

owner = pd.read_csv("owner_data.csv", header=None)
others = pd.read_csv("others_data.csv", header=None)

owner["label"] = 1
others["label"] = 0

data = pd.concat([owner, others], ignore_index=True)

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "hand_auth.pkl")

print("Model Saved Successfully")
