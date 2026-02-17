#pandas functions are basically excel inside of python, with a bunch of functionalities
import pandas as pd
#splits the  data into two parts to prevent the model from "memorizing" the data to ensure the scores are honest
from sklearn.model_selection import train_test_split
#mltiple things together, forest is many decision trees, classifier indicates the goal is to categorize and not predict an outcome
from sklearn.ensemble import RandomForestClassifier
#takes predicted results from AI and compares to the real results
from sklearn.metrics import accuracy_score

#select the open source data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
#reads in the csv from the url to the data variable and seperates each variable based on the presence of the ;
data = pd.read_csv(url, sep=';')

#make the columns
#this makes it so X is every column except quality, which is Y 
X = data.drop('quality', axis=1)
#turns quality scores 7 and 8 into a '1' (True) and lower than 7 is a '0'(False)
Y = (data['quality']>=7).astype(int)

#split the data for the train test split
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#initialize and train the model
#n_estimators is how many "trees" are used
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,Y_train)

#make predictions and compare accuray
prediction = model.predict(X_test)
score = accuracy_score(Y_test,prediction)

#print the score at the end
print(f"Model Accuray: {score * 100:.2f}%")



#determine the most import columns to improve accuracy
importance = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names,importance))
#sort and print those importances
for feature, importance in sorted(feature_importance_dict.items(), key = lambda item: item[1], reverse=True):
    print(f"{feature}: {importance:.4f}")