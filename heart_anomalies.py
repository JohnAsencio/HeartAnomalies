import numpy as np
import sys
import csv
from fractions import Fraction

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csvReader = csv.reader(file)
        for row in csvReader:
            label = int(row[0])
            features = list(map(int, row[1:]))
            data.append({'c': label, 'f': features})

   # print(data)
    return data

def naive_bayes(trainingData):
    instanceClassification = 2  
    numFeatures = len(trainingData[0]['f'])

    F = np.zeros((instanceClassification, numFeatures))
    N = np.zeros(instanceClassification)

    for instance in trainingData:
      #  print(instance)
        N[instance['c']] += 1
        for j in range(numFeatures):
            if instance['f'][j] == 1:
                F[instance['c'], j] += 1

   # print(N, F)
    return N, F

def compute_likelihood(instance, F, N):
    L = np.zeros(len(N))
# print(instance)

    for i in range(len(N)):
        L[i] = np.log(N[i] + 0.5) - np.log(N[0] + N[1] + 0.5)
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
   # PP = 0
   # PN = 0
    totP = 0
    totN = 0

    for instance in testData:
        L = compute_likelihood(instance, F, N)
        predictedClass = classify_instance(L)
        if predictedClass == instance['c']:
            correct += 1
            if instance['c'] == 0:
                trueNegative += 1
            elif instance['c'] == 1:
                truePositive += 1
        
        if instance['c'] == 0:
            totP += 1
        elif instance['c'] == 1:
            totN += 1
    

  #  print(correct, trueNegative, truePositive, len(testData))
    accuracy = correct / len(testData)
    trueNegativeRate = trueNegative / totP
    truePositiveRate = truePositive / totN

    return accuracy, trueNegativeRate, truePositiveRate

def k_fold_cross_validation(data):
    k = 5
    np.random.shuffle(data)
    foldSize = len(data) // k
    accuracies = []
    tnRates = []
    tpRates = []

    for i in range(k):
        start = i * foldSize
        end = (i + 1) * foldSize
        testData = data[start:end]
    #    print(test_data, i, '\n')
        trainingData = data[:start] + data[end:]
       # print(start, end)

    ##    print("training data: ", training_data, '\n Test: ', test_data, '\n', i, '\n')

        N, F = naive_bayes(trainingData)
        accuracy, tnRate, tpRate = evaluate_naive_bayes(testData, N, F)

        accuracies.append(accuracy)
        tnRates.append(tnRate)
        tpRates.append(tpRate)

    avgAccuracy = np.mean(accuracies)
    avgTnRate = np.mean(tnRates)
    avgTpRate = np.mean(tpRates)

    return avgAccuracy, avgTnRate, avgTpRate

def main():
    if len(sys.argv) != 2:
        print("You must input a data file")
        sys.exit(1)

    dataPath = sys.argv[1]

    data = load_data(dataPath)

    avgAccuracy, avgTnRate, avgTpRate = k_fold_cross_validation(data)

    print(f'Accuracy: {Fraction(avgAccuracy).limit_denominator(200)}({avgAccuracy:.2f}) '
          f'True Negative Rate: {Fraction(avgTnRate).limit_denominator(200)}({avgTnRate:.2f}) '
          f'True Positive Rate: {Fraction(avgTpRate).limit_denominator(200)}({avgTpRate:.2f})')


if __name__ == "__main__":
    main()