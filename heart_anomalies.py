import numpy as np
import sys
import csv
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

def naive_bayes_learner(trainingData):
    instanceClassification = 2  
    numFeatures = len(trainingData[0]['f'])

    # Initialize counts
    F = np.zeros((instanceClassification, numFeatures))
    N = np.zeros(instanceClassification)

    # Iterate over each training instance
    for instance in trainingData:
        N[instance['c']] += 1
        for j in range(numFeatures):
            if instance['f'][j] == 1:
                F[instance['c'], j] += 1

    return N, F

def compute_likelihood(instance, F, N):
    L = np.zeros(len(N))

    for i in range(len(N)):
        L[i] = np.log(N[i] + 0.5) - np.log(np.sum(N[0]) + np.sum(N[1]) + 0.5)
        for j in range(len(instance['f'])):
            s = F[i, j] 
            if instance['f'][j] == 0:
                s = N[i]-s
            L[i] += np.log(s + 0.5) - np.log(N[i] + 0.5)

    return L


def classify_instance(L):
    return 1 if L[1] > L[0] else 0

def evaluate_naive_bayes(testData, N, F):
    correct = 0
    trueNegative = 0
    truePositive = 0

    for instance in testData:
        L = compute_likelihood(instance, F, N)
        predictedClass = classify_instance(L)
        if predictedClass == instance['c']:
            correct += 1
            if instance['c'] == 0:
                trueNegative += 1
            else:
                truePositive += 1

    print(truePositive, 
          len(testData),
          trueNegative)

    accuracy = correct / len(testData)
    trueNegativeRate = trueNegative / len(testData)
    truePositiveRate = truePositive / len(testData)

    return accuracy, trueNegativeRate, truePositiveRate

def main():
    if len(sys.argv) != 3:
        print("You must input a test file and a training file")
        sys.exit(1)

    trainingDataPath = sys.argv[1]
    testDataPath = sys.argv[2]

    # Load data from CSV files
    trainingData = load_data(trainingDataPath)
    testData = load_data(testDataPath)

    N, F = naive_bayes_learner(trainingData)

    # Evaluate the model on the test set
    accuracy, trueNegativeRate, truePositiveRate = evaluate_naive_bayes(testData, N, F)

    # Print the results
    print(f'orig {Fraction(accuracy).limit_denominator(200)}({accuracy:.2f}) '
          f'{Fraction(trueNegativeRate).limit_denominator(200)}({trueNegativeRate:.2f}) '
          f'{Fraction(truePositiveRate).limit_denominator(200)}({truePositiveRate:.2f})')

if __name__ == "__main__":
    main()