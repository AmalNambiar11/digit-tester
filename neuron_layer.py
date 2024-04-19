import numpy as np

class NeuronLayer:
    def __init__(self, rows, cols, extentW=2.0, extentB=2.0) -> None:
        self.rows = rows # outputs
        self.cols = cols # inputs
        self.weight = np.zeros([rows, cols])
        self.bias = np.zeros(rows)
        self.input_ = np.zeros(cols)
        self.actvn = np.zeros(rows)
        self.output_ = np.zeros(rows)

        self.memory = 0
        self.w_change = np.zeros([rows, cols])
        self.b_change = np.zeros(rows)

        self.extentW = extentW
        self.extentB = extentB
        self.randomize()

    def randomize(self, W=0.0, B=0.0):
        if (W == 0.0):
            W = self.extentW
        if (B == 0.0):
            B = self.extentB
        #print(W)
        #print(B)
        self.weight = W*(np.random.rand(self.rows, self.cols) - 0.5)
        self.bias = B*(np.random.rand(self.rows) - 0.5)

    def send(self, data, is_training=False):
        self.input_ = data
        self.actvn = np.matmul(self.weight, data) + self.bias
        self.output_ = self.sigmoid(self.actvn)
        #self.output_d = self.sigmoid_d(self.actvn)
        self.output_d = self.output_*(1-self.output_)
    
    def back(self, cost):
        self.memory += 1
        x = np.multiply(cost, self.output_d)
        self.b_change -= x
        self.w_change -= np.tensordot(x, self.input_, axes=0)
        return np.matmul(np.transpose(self.weight), x)

    def reset(self):
        self.memory = 0 
        self.w_change.fill(0)
        self.b_change.fill(0)

    def update(self, x=0.1):
        if self.memory == 0:
            return
        self.weight += x*self.w_change/self.memory
        self.bias += x*self.b_change/self.memory
        self.reset()

    def sigmoid(self, x):
        return 1/(1+ np.exp(-x))
    
#    def sigmoid_d(self, x):
#        return (np.exp(-x))/((np.exp(-x)+1)**2)
