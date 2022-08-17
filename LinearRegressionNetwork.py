from Node import Node
from Connection import Connection
from Layer import Layer
from time import time
import random

class LinearRegressionNetwork:
    def __init__(self, shape, learningRate, initWeights=None):
        self.m_shape = shape
        self.m_learningRate = learningRate
        self.m_initWeights = initWeights
        self.m_layers = []
        self.m_nodes = []
        self.m_connectionsDict = {} #toNode, fromNode
        self.m_connections = []
        self.initNetwork()

    def addLayer(self, layer):
        self.m_layers.append(layer)

    def addNode(self, node):
        self.m_nodes.append(node)

    def addConnection(self, connection):
        toNode = connection.getToNode()
        fromNode = connection.getFromNode()
        try:
            self.m_connectionsDict[toNode][fromNode] = connection
        except:
            self.m_connectionsDict[toNode] = {}
            self.m_connectionsDict[toNode][fromNode] = connection
        self.m_connections.append(connection)

    def createNodesAndLayers(self):
        for shapeIndex, shape in enumerate(self.m_shape):
            currentLayer = Layer([], self, shapeIndex, shape)
            for nodeIndex in range(0, shape):
                currentNode = Node(currentLayer, self, nodeIndex)
                currentLayer.addNode(currentNode)
                self.addNode(currentNode)
            if(shapeIndex < len(self.m_shape) - 1):
                biasNode = Node(currentLayer,self,shape,True)
                self.addNode(biasNode)
                currentLayer.addNode(biasNode)
            self.addLayer(currentLayer)

    def createConnections(self):
        random.seed(time())
        allLayersButLast = self.m_layers[:-1]
        for layerIndex, currentLayer in enumerate(allLayersButLast):
            nextLayer = self.m_layers[layerIndex + 1]
            for fromNode in currentLayer.m_nodes:
                for toNode in nextLayer.m_nodes:
                    if (toNode.isBias() == False):
                        currentConnection = Connection(fromNode, toNode)
                        if (self.m_initWeights != None):
                            currentConnection.setWeight(self.m_initWeights)
                        else:
                            currentConnection.setWeight(random.random())
                        self.addConnection(currentConnection)
                        fromNode.addConnectionUp(currentConnection)
                        toNode.addConnectionDown(currentConnection)

    def initNetwork(self):
        self.createNodesAndLayers()
        self.createConnections()

    def getInputSize(self):
        return self.m_shape[0]

    def getOutputSize(self):
        return self.m_shape[-1]

    def getLayer(self, index=-1):
        return self.m_layers[index]

    def getLearningRate(self):
        return self.m_learningRate

    def setLearningRate(self, learningRate):
        self.m_learningRate = learningRate

    def getConnectionsDict(self):
        return self.m_connectionsDict

    def getConnections(self):
        return self.m_connections

    def getNodes(self):
        return self.m_nodes

    def setInputLayer(self, input):
        for i, node in enumerate(self.getLayer(0).getNodes()):
            if (node.isBias() == False):
                node.setValue(input[i])
            else:
                node.setValue(1)

    def forward(self, input):
        if (len(input) != self.getInputSize()):
            raise IOError("Input is not in the right size")
        self.setInputLayer(input)
        for previousLayerIndex, currentLayer in enumerate(self.m_layers[1: ]):
            for toNode in currentLayer.getNodes():
                if(toNode.isBias() == False):
                    newValue = 0
                    for fromNode in self.m_connectionsDict[toNode]:
                        connection = self.m_connectionsDict[toNode][fromNode]
                        newValue += connection.getWeight() * fromNode.getValue()
                    toNode.setValue(newValue)
        lastLayer = self.getLayer(-1)
        return lastLayer.getLayerInArrayFormat()

    def calculateLoss(self, expectedOutput):
        if (self.getOutputSize() != len(expectedOutput)):
            raise IOError(f"Output is not in the right size. Output: {self.getOutputSize()}, Expected output: {expectedOutput}")
        lastLayer = self.getLayer(-1)
        lastLayerArray = lastLayer.getLayerInArrayFormat()
        loss = 0
        for index, value in enumerate(lastLayerArray):
            expectedValue = expectedOutput[index]
            loss += (value - expectedValue) ** 2
        return loss

    def calculateGradientForOutputLayer(self, expectedOutput):
        if (self.getOutputSize() != len(expectedOutput)):
            raise IOError("Output is not in the right size")
        lastLayer = self.getLayer(-1)
        lastLayerArray = lastLayer.getLayerInArrayFormat()
        for index, value in enumerate(lastLayerArray):
            expectedValue = expectedOutput[index]
            currentNode = lastLayer.getNodes()[index]

            #The dLoss / dLastLayerNodes =  2 * (value - expectedValue)
            value = 2 * (value - expectedValue)
            value /= self.getOutputSize()
            currentNode.setGradient(value)

    def backwardNodeGradient(self):
        # dLoss/dFromNode = SUM((dLoss/dToNode) * (dToNode/dFromNode))
        # dToNode/dFromNode = Weight between nodes
        allLayersButLast = self.m_layers[:-1]
        for layer in allLayersButLast[::-1]:
            for fromNode in layer.getNodes():
                if (fromNode.isBias() == False):
                    sum = 0
                    for connectionUp in fromNode.getConnectionsUp():
                        toNode = connectionUp.getToNode()
                        sum += toNode.getGradient() * connectionUp.getWeight()
                    sum /= layer.getLength()
                    fromNode.setGradient(sum)

    def updateWeightsGradient(self):
        #dLoss/dWeight = dLoss/dToNode * dToNode/dFromNode
        for connection in self.getConnections():
            toNode = connection.getToNode()
            fromNode = connection.getFromNode()
            gradientValue = toNode.getGradient() * fromNode.getValue()
            connection.setGradient(gradientValue)

    def step(self):
        for connection in self.getConnections():
            currentWeight = connection.getWeight()
            currentWeight -= connection.getGradient() * self.getLearningRate()
            connection.setWeight(currentWeight)

    def train(self, input, expectedOutput):
        self.forward(input)
        self.calculateGradientForOutputLayer(expectedOutput)
        self.backwardNodeGradient()
        self.updateWeightsGradient()
        self.step()