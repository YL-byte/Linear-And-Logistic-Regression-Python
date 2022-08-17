class Connection:
    def __init__(self, fromNode, toNode):
        self.m_fromNode = fromNode
        self.m_toNode = toNode
        self.m_weight = 0
        self.m_gradient = 0

    def getFromNode(self):
        return self.m_fromNode

    def getToNode(self):
        return self.m_toNode

    def setWeight(self, value):
        self.m_weight = value

    def getWeight(self):
        return self.m_weight

    def getGradient(self):
        return self.m_gradient

    def setGradient(self, value):
        self.m_gradient = value