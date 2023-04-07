import csv
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Read training data from a CSV file
training_data = []
with open('training_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        text = row['Text']
        regex = row['Regex']
        training_data.append((text, regex))

# Preprocess the training data
train_text = []
train_labels = []
for data in training_data:
    train_text.append(data[0])
    train_labels.append(data[1])

# Vectorize the training data
vectorizer = CountVectorizer()
train_text_vectors = vectorizer.fit_transform(train_text)

# Train the model
clf = MultinomialNB()
clf.fit(train_text_vectors, train_labels)

# Take user input and make a prediction
input_text = input("Enter some text: ")
input_text_vector = vectorizer.transform([input_text])
predicted_label = clf.predict(input_text_vector)

# Print the predicted regular expression
print("Predicted regex pattern:", predicted_label[0])
