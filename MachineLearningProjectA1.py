
#sklearn is a prebuilt python library 
from sklearn.datasets import load_iris  #load_iris is a built in dataset
from sklearn.model_selection import train_test_split  #split data into training and testing data sets
from sklearn.neighbors import KNeighborsClassifier  #KNeighbors classifier is a machinelearning algorithm
from sklearn.metrics import accuracy_score #accuracy score gradse the predictions

# Load dataset
iris = load_iris()
X = iris.data      # features
y = iris.target    # labels

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)