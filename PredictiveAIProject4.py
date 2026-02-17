#pandas functions are basically excel inside of python, with a bunch of functionalities
import pandas as pd
#splits the  data into two parts to prevent the model from "memorizing" the data to ensure the scores are honest
from sklearn.model_selection import train_test_split
#mltiple things together, forest is many decision trees, classifier indicates the goal is to categorize and not predict an outcome
from sklearn.ensemble import RandomForestClassifier
#takes predicted results from AI and compares to the real results
from sklearn.metrics import accuracy_score
# breaks down AI guesses to 4 categories to show the results and then makes it a statistical score
from sklearn.metrics import confusion_matrix,classification_report
#matplotlib is used to graph everything
import matplotlib.pyplot as plt
#seaborn is a specialized python tool to create statistical graphs
import seaborn as sns
#tool to save and load python objects with large number arrays
import joblib

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
model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
model.fit(X_train,Y_train)

#make predictions and compare accuray
prediction = model.predict(X_test)
score = accuracy_score(Y_test,prediction)

#print the score at the end
print(f"Model Accuray: {score * 100:.2f}%")

#show the confusion matrix and detailed report
print("\n--- Confusion Matrix ---")
print(confusion_matrix(Y_test, prediction))

# Shows Precision and Recall (how good it is at finding 'Good' wine)
print("\n--- Detailed Report ---")
print(classification_report(Y_test, prediction))

#determine the most import columns to improve accuracy
importance = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names,importance))
#sort and print those importances
for feature, importance in sorted(feature_importance_dict.items(), key = lambda item: item[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

#add visualization of the result
#make the data
CM = confusion_matrix(Y_test,prediction)

#make the graph
plt.figure(figsize=(8,6))

#make heatmap of the graph
sns.heatmap(CM,annot=True,fmt='d',cmap='Blues', xticklabels=['Predicted Average', 'Predicted Good'],yticklabels=['Actual Average','Actual Good'])

#set up labels
plt.xlabel('AI Predictions')
plt.ylabel('Reality(Actual Labels)')

#name the graph
plt.title('Wine Quality: Confusion Matrix Heatmap')

#show the graph
plt.show()

#save the model
joblib.dump(model,'wine_prediction_model.pkl')
#state the model is saved
print("Model Saved")