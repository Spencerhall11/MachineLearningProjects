
from ucimlrepo import fetch_ucirepo
#pandas functions are basically excel inside of python, with a bunch of functionalities
import pandas as pd
#
import xgboost as xgb
#
from sklearn.model_selection import train_test_split
#
from sklearn.metrics import accuracy_score, classification_report


#fetch data
print("Fetching data from UCI")
#pulls the adult data set
adult = fetch_ucirepo(id=2)
#collect data as pandas dataframes
X = adult.data.features
Y = adult.data.targets

#pre-processing
df = pd.concat([X,Y],axis=1).replace('?',pd.NA).dropna()

#split and clean labels (some labels end in . )
y_clean = df['income'].astype(str).str.replace('.', '', regex=False).str.strip()
y_encoded = y_clean.map({'<=50K': 0, '>50K': 1})

# Separate features and use One-Hot Encoding for categorical columns
X_clean = df.drop('income', axis=1)
X_encoded = pd.get_dummies(X_clean)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
    )

# Initialize and Train XGBoost
# binary:logistic is the standard objective for 0/1 classification
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42
    )
#state that it is training and train it
print("Training XGBoost model...")
model.fit(X_train, y_train)

#Evaluate results
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))