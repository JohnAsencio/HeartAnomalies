import sys
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from fractions import Fraction

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            label = int(row[0])
            features = list(map(int, row[1:]))
            data.append({'c': label, 'f': features})
    return data

def train_and_evaluate_sklearn(training_data, test_data):
    X_train = [instance['f'] for instance in training_data]
    y_train = [instance['c'] for instance in training_data]
    
    X_test = [instance['f'] for instance in test_data]
    y_test = [instance['c'] for instance in test_data]

    # Train the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    true_negative_rate = tn / (tn + fp)
    true_positive_rate = tp / (tp + fn)

    return accuracy, true_negative_rate, true_positive_rate

def main():
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py training_data_file.csv test_data_file.csv")
        sys.exit(1)

    training_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    # Load data from CSV files
    training_data = load_data(training_data_path)
    test_data = load_data(test_data_path)

    # Train and evaluate the model
    accuracy, trueNegativeRate, truePositiveRate = train_and_evaluate_sklearn(training_data, test_data)

    # Print the results
    print(f'orig {Fraction(accuracy).limit_denominator(200)}({accuracy:.2f}) '
          f'{Fraction(trueNegativeRate).limit_denominator(200)}({trueNegativeRate:.2f}) '
          f'{Fraction(truePositiveRate).limit_denominator(200)}({truePositiveRate:.2f})') 

if __name__ == "__main__":
    main()
