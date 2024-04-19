import numpy as np
import neuron_layer as nl

class ClassifierBot:
    def __init__(self, *args):
        self.dims = []
        self.layers = []
        n = len(args)
        if (n==0):
            return
        for i in range(n):
            self.dims.append(args[i])
            if (i==0): 
                continue
            self.layers.append(nl.NeuronLayer(args[i], args[i-1]))
            self.layers[i-1].randomize()
        self.dims.append(10)
        self.layers.append(nl.NeuronLayer(10, args[n-1]))
        self.layers[n-1].randomize()
    
    def process_image(self, data, label, is_training=False):
        # takes in image data and processes it until it reaches a final array of 10 numbers
        data = data.reshape(784).astype(np.float32)
        data = data / 255.0

        n = len(self.dims)-1

        self.layers[0].send(data)
        out = []
        for i in range(n):
            out = self.layers[i].output_
            if (i==n-1):
                break
            self.layers[i+1].send(out)
        
        if is_training:
            cost = np.zeros(10)
            cost[label] = 1
            cost = 2*(out - cost)

            x = self.layers[n-1].back(cost)
            for i in range(1, n):
                x = self.layers[n-i-1].back(x)

        return out
    
    def predict_digit(self, data):
        # takes in an array of 10 numbers and gives digit with highest activation
        return np.argmax(data)
    
    def modify_bot(self, x=0.1):
        n = len(self.dims)-1
        for i in range(n):
            self.layers[i].update(x)
    
    def load_file(self, model):
        filename = "models/"+model+".pkl"
        with open(filename, 'rb') as f:
            self.dims = pickle.load(f)
            n = len(self.dims) - 1
            for i in range(n):
                new_layer = pickle.load(f)
                self.layers.append(new_layer)
                new_layer.reset()
            f.close()
    
    def save_file(self, model):
        filename = "models/"+model+".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.dims, f)
            n = len(self.dims) - 1
            for i in range(n):
                pickle.dump(self.layers[i], f)
            f.close()


