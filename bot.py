import numpy as np
import neuron_layer as nl
import pickle

class ClassifierBot:
    def __init__(self, x=2.0, is_randomized=False):
        self.n0 = nl.NeuronLayer(16, 784)
        self.n1 = nl.NeuronLayer(16, 16) 
        self.n2 = nl.NeuronLayer(10, 16)
        if is_randomized:
            self.n0.randomize(x, x)
            self.n1.randomize(x, x)
            self.n2.randomize(x, x) 
    
    def test_image(self, data, label):
        pass

    def train_image(self, data, label):
        pass

    def process_image(self, data, label, is_training=False):
        # takes in image data and processes it until it reaches a final array of 10 numbers
        data = data.reshape(784).astype(np.float32)
        data = data / 255.0
        self.n0.send(data) # input image data
        self.n1.send(self.n0.output_)
        self.n2.send(self.n1.output_)
        if is_training:
            cost = np.zeros(10)
            cost[label] = 1
            cost = 2*(self.n2.output_ - cost)
            x = self.n2.back(cost)
            y = self.n1.back(x)
            z = self.n0.back(y)
        return self.n2.output_
    
    def predict_digit(self, data):
        # takes in an array of 10 numbers and gives digit with highest activation
        return np.argmax(data)
    
    def modify_bot(self, x=0.1):
        self.n0.update(x)
        self.n1.update(x)
        self.n2.update(x)
    
    def load_file(self, model_num=1):
        filename = "models/model"+str(model_num)+".pkl"
        with open(filename, 'rb') as f:
            self.n0.weight = pickle.load(f)
            self.n1.weight = pickle.load(f)
            self.n2.weight = pickle.load(f)
            self.n0.bias = pickle.load(f)
            self.n1.bias = pickle.load(f)
            self.n2.bias = pickle.load(f)
            f.close() 

    def save_file(self, model_num=1):
        filename = "models/model"+str(model_num)+".pkl"
        with open(filename, 'wb') as f:  
            pickle.dump(self.n0.weight, f)
            pickle.dump(self.n1.weight, f)
            pickle.dump(self.n2.weight, f)
            pickle.dump(self.n0.bias, f)
            pickle.dump(self.n1.bias, f)
            pickle.dump(self.n2.bias, f)
            f.close()