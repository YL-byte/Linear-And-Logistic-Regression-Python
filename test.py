from LinearRegressionNetwork import LinearRegressionNetwork
from LogisticRegressionNetwork import LogisticRegressionNetwork
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
import pickle

def test1(save=False):
    nn = LinearRegressionNetwork(shape=[2, 4, 2], learningRate=0.1)
    for connection in nn.getConnections():
        print(connection.getFromNode().isBias(), connection.getWeight())

    excel_file = pd.ExcelFile("Linear Regression Model.xlsx")
    ds1 = pd.read_excel(excel_file, "Example1")
    data = ds1.values
    for item in data:
        input = item[0:2]
        output = item[2:4]
        nn.train(input, output)
    for connection in nn.getConnections():
        print(connection.getFromNode().isBias(), connection.getWeight())
    # for node in nn.getNodes():
    #     print(node.isBias(), node.getGradient())
    if (save):
        pickle.dump(nn, open("data/test1save_" + str(time()) + ".p", "wb"))
    print(nn.forward([1,1]))

def test2(save=False):
    nn = LinearRegressionNetwork([1, 1], learningRate = 0.1)
    excel_file = pd.ExcelFile("Linear Regression Model.xlsx")
    ds1 = pd.read_excel(excel_file, "Example2")
    data = ds1.values
    for item in data:
        input = [item[0]]
        output = [item[1]]
        nn.train(input, output)
    for connection in nn.getConnections():
        print(connection.getFromNode().isBias(), connection.getWeight())
    if(save):
        pickle.dump(nn, open("data/test2save_" + str(time()) + ".p", "wb"))

def test3(save=False):

    nn = LogisticRegressionNetwork(shape=[784, 10], learningRate=0.1, initWeights=0.1)
    MAX_PIXEL = 256
    loss = []
    with open("mnist_train.txt", 'r') as f:
        rawData = f.read().split('\n')
    for rawInputIndex, rawInput in enumerate(rawData):
        try:
            expectedOutput = [0] * 10
            input = rawInput.split(',')[1:]
            for index, element in enumerate(input):
                input[index] = int(element) / MAX_PIXEL
            expectedOutput[int(rawInput.split(',')[0])] = 1
            nn.train(input, expectedOutput)
            loss.append(nn.calculateLoss(expectedOutput))
        except:
            print("Training - Error in index",rawInputIndex)
    # plt.plot(loss)
    # plt.show()
    if(save):
        pickle.dump(nn, open("data/test3save_" + str(time()) + ".p", "wb"))
        pickle.dump(nn, open("data/test3save" + ".p", "wb"))
    success = 0
    failure = 0

    with open("mnist_test.txt", 'r') as f:
        rawData = f.read().split('\n')

    for rawInputIndex, rawInput in enumerate(rawData):
        try:
            expectedOutput = [0] * 10
            input = rawInput.split(',')[1:]
            for index, element in enumerate(input):
                input[index] = int(element) / MAX_PIXEL
            expectedOutput[int(rawInput.split(',')[0])] = 1
            output = nn.forward(input)
            chosenOutput = 0
            chosenResult = output[0]
            for output, result in enumerate(output):
                if (result > chosenResult):
                    chosenOutput = output
                    chosenResult = result

            for output, result in enumerate(expectedOutput):
                if result == 1:
                    expectedOutput = output

            # print("="*20)
            # print("Expected Output:", expectedOutput)
            # print("Output:", chosenOutput)
            if(expectedOutput == chosenOutput):
                # print("Yes")
                success += 1
            else:
                # print("No")
                failure += 1

        except:
            print("Executing - Error in index",rawInputIndex)
    print("Success:", success, "| Failure:", failure)

if __name__ == "__main__":
    # startTime = time()
    # test3(save=True)
    # print("Finished in", time() - startTime, "Seconds")
    nn = pickle.load(open("data/test3save.p", "rb"))

    success = 0
    failure = 0
    MAX_PIXEL = 256
    with open("mnist_test.txt", 'r') as f:
        rawData = f.read().split('\n')
    successHist = [0] * 10
    totalHist = [0] * 10
    for rawInputIndex, rawInput in enumerate(rawData):
        expectedOutput = [0] * 10
        input = rawInput.split(',')[1:]
        for index, element in enumerate(input):
            input[index] = int(element) / MAX_PIXEL
        expectedOutput[int(rawInput.split(',')[0])] = 1
        output = nn.forward(input)
        chosenOutput = 0
        chosenResult = output[0]
        for output, result in enumerate(output):
            if (result > chosenResult):
                chosenOutput = output
                chosenResult = result

        for output, result in enumerate(expectedOutput):
            if result == 1:
                expectedOutput = output

        totalHist[expectedOutput] += 1
        if(expectedOutput == chosenOutput):
            success += 1
            successHist[expectedOutput] += 1
        else:
            failure += 1

    for i in range(0, 10):
        print(i, totalHist[i], successHist[i], successHist[i]/ totalHist[i])
    print("Success:", success, "| Failure:", failure, "| Success Rate:", success / (success + failure))