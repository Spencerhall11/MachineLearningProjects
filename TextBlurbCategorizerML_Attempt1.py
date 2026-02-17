#Goal: Take a text blurb and categorize it
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Inputs
text_Entries = [
    "The team won the championship game",
    "The match went into overtime",
    "The player broke the record with three touchdowns",
    "The government passed a new law",
    "Parliament debated the budget",
    "The president signed a new bill",
    "The latest smartphone was released",
    "New AI software beats benchmarks",
    "Researchers unveiled a new programming language",
    "The actor starred in a new movie",
    "The film won several awards",
    "The director released a documentary"
]

#Categories
text_Labels = [
    "Sports", "Sports", "Sports",
    "Politics", "Politics", "Politics",
    "Technology", "Technology", "Technology",
    "Entertainment", "Entertainment", "Entertainment"
]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_Entries)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, text_Labels, test_size=0.25, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Try your own text
while True:
    text = input("\nEnter a blurb (or 'quit'): ")
    if text.lower() == "quit":
        break

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    print("Predicted category:", prediction[0])