import matplotlib.pyplot as plt
import numpy as np

class GraphTool:
    def __init__(self, size=1):
        self.X = np.arange(1, size+1)
        self.Y = np.empty(size)
        self.N = 0

    def add_point(self, correct, total=1500):
        self.Y[self.N] = correct/total*100
        self.N += 1
    
    def recent_avg(self):
        data = 0.0
        N = self.N
        for i in range(N-1, N-1-20, -1):
            data += self.Y[i]
        data = data/20
        print("Average success rate in last 20 trials: ", data)
    
    def draw_graph(self):
        plt.title("Digit Tester")  
        plt.plot(self.X, self.Y)
        self.recent_avg()
        plt.ylabel('% Accuracy')
        plt.xlabel('Batches')
        plt.show()