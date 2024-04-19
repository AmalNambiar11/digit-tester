from keras.datasets import mnist
from random import randrange
import numpy as np 
from bot2 import ClassifierBot
from graph_tool import GraphTool

print("Loading MNIST data ..")
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
print("Loaded")

TRAINING_SET_SIZE = len(train_X) #60000
TESTING_SET_SIZE = len(test_X) #10000


def test_single_image(bot, x, is_training=True):
    data = train_X[x]
    label = train_Y[x]
    y = bot.process_image(data, label, is_training)
    z = bot.predict_digit(y)
    print("Image randomly selected: ", x+1)
    print("Label is: ", label)
    print("Result from bot is: ", y)
    print("The digit predicted is: ", z)

def test_in_order(bot, n, is_detailed=False, graph=None, is_training=True):
    totalCorrect = 0
    for x in range(n):
        data = train_X[x]
        label = train_Y[x]
        y = bot.process_image(data, label, is_training)
        z = bot.predict_digit(y)
        if is_detailed:
            print("Image randomly selected: ", x)
            print("Label is: ", label)
            print("Result from bot is: ", y)
            print("The digit predicted is: ", z)
        if z == label:
            totalCorrect += 1
    if not graph == None:
        graph.add_point(totalCorrect, n)
    print("Total correct tests: ", totalCorrect, " out of ", n)

def test_batch(bot, n, is_detailed=False, graph=None, is_training=True):
    totalCorrect = 0
    for i in range(n):
        x = randrange(0, TRAINING_SET_SIZE)
        data = train_X[x]
        label = train_Y[x]
        y = bot.process_image(data, label, is_training)
        z = bot.predict_digit(y)
        if is_detailed:
            print("Image randomly selected: ", x)
            print("Label is: ", label)
            print("Result from bot is: ", y)
            print("The digit predicted is: ", z)
        if z == label:
            totalCorrect += 1
    if not graph == None:
        graph.add_point(totalCorrect, n) 
    print("Total correct tests: ", totalCorrect, " out of ", n)

bot1 = ClassifierBot(784, 16, 16)#2.0, True)

def start():
    trials = 150
    rate = 0.2
    g = GraphTool(trials)
    for i in range(trials):
        print("Trial ", i+1, "- ", end="")
        test_batch(bot1, 1500, False, g)
        bot1.modify_bot(rate)
        if i>400:
            rate = 0.5
    g.draw_graph()

def start2():
    trials = 500
    rate = 1.0#0.1
    g = GraphTool(trials)
    for i in range(trials):
        print("Trial ", i+1, "- ", end="")
        test_batch(bot1, 1500, False, g)
        bot1.modify_bot(rate)
        if i>400:
            rate = 1.0
    #bot1.save_file(2)
    g.draw_graph()

start2()


    