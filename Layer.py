from Node import Node
class Layer:
    def __init__(self, nodes, network, index, length):
        self.m_nodes = nodes
        self.m_network = network
        self.m_index = index
        self.m_length = length

    def addNode(self, node):
        self.m_nodes.append(node)

    def getNodes(self):
        return self.m_nodes

    def getLayerInArrayFormat(self):
        arr = []
        for node in self.getNodes():
            if (node.isBias() == False):
                value = node.getValue()
                arr.append(value)
        return arr

    def getLength(self):
        return self.m_length