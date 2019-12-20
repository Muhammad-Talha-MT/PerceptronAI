import pandas as pd
import numpy as num
import sys


def getAccureccy(correct, total):  # find Accureccy
    return correct/total*100


def testData(testFile, trainedFile):  # test Data
    pd_learning = pd.read_csv(testFile)
    rows, _ = pd_learning.shape
    global bised
    pd_learned = pd.read_csv(trainedFile)
    final = list(pd_learned.iloc[-1, 7:11])
    correct = 0

    for i in range(rows):
        inputFeatures = list(pd_learning.iloc[i, :])
        actual = inputFeatures.pop()
        predicted = predict(inputFeatures, final)
        print('Actual', actual, 'Predicted', predicted)
        if actual == predicted:
            correct = correct + 1
    accureccy = getAccureccy(correct, rows)
    print('----------')
    print('Accureccy: ', accureccy, '%')


def checkError(errorlist):  # check error
    if all(v == 0 for v in errorlist):
        return False
    else:
        return True


def predict(input, weight):  # predict model
    global bised
    z = 0
    for i in range(len(input)):
        z = z + input[i]*weight[i]
    if z >= bised:
        return 1
    else:
        return 0


def updateWeight(yd, yp, input):  # updatae weights
    global weights
    a = 0.1
    for i in range(len(weights)):
        weights[i] = weights[i]+a*(yd-yp)*input[i]


def saveEpoch(epoch, input, weights, predictedValue, actualValue, error):  # save each epoch in data
    global pd_epoch
    dic = {'epoch': epoch, 'x1': input[0], 'x2': input[1], 'x3': input[2], 'x4': input[3], 'y': predictedValue, 'w0': bised,
           'w1': weights[0], 'w2': weights[1], 'w3': weights[2], 'w4': weights[3], 'd': actualValue, 'error': error}

    pd_epoch = pd_epoch.append(dic, ignore_index=True)


def learning():  # learning perceptron
    global pd_epoch
    global weights
    global bised
    global pd_learning
    global columns
    global inputs
    isError = True
    epoch = 1
    while(isError):
        errorList = []
        print("Episode #", epoch)
        print('[', weights[0], weights[1], weights[2], weights[3], ']')
        for i in range(rows):
            input = list()
            for j in range(columns):
                input.append(pd_learning.loc[i, inputs[j]])
            actualValue = input.pop()
            predictedValue = predict(input, weights)

            error = actualValue - predictedValue
            errorList.append(error)
            saveEpoch(epoch, input, weights,
                      predictedValue, actualValue, error)

            updateWeight(actualValue, predictedValue, input)

        epoch = epoch + 1

        isError = checkError(errorList)
    pd_epoch.to_csv('learnedData.csv', index=False)
    print("--------------")
    print('Total Episodes', epoch-1)
    print('Final Weights', weights, 'Treshold:', bised)


if __name__ == "__main__":
    command = sys.argv
    col = ['epoch', 'x1', 'x2', 'x3', 'x4', 'y', 'w0',
           'w1', 'w2', 'w3', 'w4', 'd', 'error']
    pd_epoch = pd.DataFrame(columns=col)

    pd_learning = pd.read_csv(command[2])
    rows, columns = pd_learning.shape
    inputs = list(pd_learning.columns)
    weights = [2.1]*(len(inputs)-1)
    bised = 2.3
    if command[1] == '--train':
        learning()
    elif command[1] == '--test':
        testData(command[2], command[4])
