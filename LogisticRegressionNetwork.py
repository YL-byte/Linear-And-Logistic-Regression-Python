from Node import Node
from Connection import Connection
from Layer import Layer
from LinearRegressionNetwork import LinearRegressionNetwork
import numpy as np

class LogisticRegressionNetwork(LinearRegressionNetwork):
    @staticmethod
    def sigmoid(x):
        result = 1 + np.e ** (-1 * x)
        result = 1 / result
        return result

    @staticmethod
    def sigmoidDerivative(x):
        return LogisticRegressionNetwork.sigmoid(x)*LogisticRegressionNetwork.sigmoid(1-x)

    def __init__(self, shape, learningRate, initWeights=None, activationFunction=None, activationFunctionDerivative=None):
        LinearRegressionNetwork.__init__(self, shape, learningRate, initWeights)
        if activationFunction == None:
            self.m_activationFunction = LogisticRegressionNetwork.sigmoid
        else:
            self.m_activationFunction = activationFunction
        if activationFunctionDerivative == None:
            self.m_activationFunctionDerivative = LogisticRegressionNetwork.sigmoidDerivative
        else:
            self.m_activationFunctionDerivative = activationFunctionDerivative
        self.setAllNodesLogistic()
        self.setActivationDerivative()

    def setAllNodesLogistic(self):
        for node in self.getNodes():
            if (node.getLayer() != self.m_layers[0]):
                node.setIsLogistic(True)

    def setActivationDerivative(self):
        for node in self.getNodes():
            if (node.getLayer() != self.m_layers[0]):
                node.setActivationFunction(self.m_activationFunction)
                node.setActivationFunctionDerivative(self.m_activationFunctionDerivative)

    def calculateGradientForOutputLayer(self, expectedOutput):
        if (self.getOutputSize() != len(expectedOutput)):
            raise IOError("Output is not in the right size")
        lastLayer = self.getLayer(-1)
        lastLayerArray = lastLayer.getLayerInArrayFormat()
        for index, value in enumerate(lastLayerArray):
            expectedValue = expectedOutput[index]
            currentNode = lastLayer.getNodes()[index]

            #The dLoss / dLastLayerNodes =  2 * (value - expectedValue)
            newValue = 2 * (value - expectedValue) * self.m_activationFunctionDerivative(value)
            newValue /= self.getOutputSize()
            currentNode.setGradient(newValue)

    def backwardNodeGradient(self):
        # dLoss/dFromNode = SUM((dLoss/dToNode) * (dToNode/dSigmoid) * (dSigmoid/dFromNode)
        # dToNode/dFromNode = Weight between nodes
        allLayersButLastAndFirst = self.m_layers[:-1]
        for layer in allLayersButLastAndFirst[::-1]:
            for fromNode in layer.getNodes():
                if (fromNode.isBias() == False):
                    sum = 0
                    for connectionUp in fromNode.getConnectionsUp():
                        toNode = connectionUp.getToNode()
                        sum += toNode.getGradient() * self.m_activationFunctionDerivative(toNode.getValue()) * connectionUp.getWeight()
                    sum /= layer.getLength()
                    fromNode.setGradient(sum)

    def updateWeightsGradient(self):
        # dLoss/dWeight = dLoss/dToNode * dToNode/dSigmoid * dSigmoid/dFromNode
        for connection in self.getConnections():
            toNode = connection.getToNode()
            fromNode = connection.getFromNode()
            gradientValue = toNode.getGradient() * self.m_activationFunctionDerivative(toNode.getValue()) * fromNode.getValue()
            connection.setGradient(gradientValue)

