import numpy as np
from numpy import ndarray
import pandas as pd
import sklearn
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import random


class MainOperation():
    def __init__(self):
        self.dataset = 'inflationData.csv'
        # creating my "entryway" into the other classes
        self.function = ActivationFunctions()
        self.data = DatasetPreperation()
        self.network = NeuralNetwork()
        # General
        self.epochs = 1
        self.learningRate = 0.01
        self.valiFreq = 1

        self.currentBatch = 0

    def main(self, dataset):
        # need to feedforward, backprop, get gradients, an store in dataset for every sample until batch is completed, then train
        preprocessedData = self.data.preprocess(dataset)
        batches = self.data.createBatches(preprocessedData)
        self.network.createNetwork(batches)
        # start training
        for epoch in range(self.epochs):
            np.random.shuffle(self.data.trainBatches)
            self.currentBatch = 0
            for batch in range(self.data.trainBatchCount):
                if epoch % 10 == 0 and batch == 0:
                    valiBatch = self.network.pickValiBatch(batches)
                    valiOutput = self.network.feedForward(valiBatch)

                trainBatch = self.network.pickTrainBatch(batches)
                trainOutput = self.network.feedForward(trainBatch)
                self.network.backProp(trainOutput[0], trainOutput[1])

                self.currentBatch += 1

class DatasetPreperation():
    def __init__(self):
        # Batches
        self.batchSize = 64
        self.trainingSplit = 0.8

    def preprocess(self, dataset):
        print("Preprocessing Data...")
        # bring in inflation csv and turn into numpy array
        data = pd.read_csv(dataset).to_numpy()
        # deletes the first row because they weren't actual data
        # and then the first column because those are row labels
        data = np.delete(data, 0, axis=0)
        data = np.delete(data, 0, axis=1)
        # scaling down all of the data to small numbers
        scaledData = RobustScaler().fit_transform(data)

        return scaledData

    def createBatches(self, data):
        print("\nCreating Batches...\n")
        # figure out how many rows will be for training and how many for validation
        trainSize = int(data.shape[0] * self.trainingSplit)
        valiSize = int(data.shape[0] * (1 - self.trainingSplit))
        # initialize training and validation batches
        trainBatchCount = int(trainSize / self.batchSize)
        self.trainBatchCount = trainBatchCount  # to use in mainOp
        self.trainBatches = ['placeholder'] * trainBatchCount
        valiBatchCount = int(valiSize / self.batchSize)
        valiBatches = ['placeholder'] * valiBatchCount
        valiData = data[trainSize: (trainSize + valiSize), 0:7]
        # fill in the training batches with actual data
        for batch in range(trainBatchCount):
            self.trainBatches[batch] = data[(batch * self.batchSize): ((batch + 1) * self.batchSize), 0:7]
        print("\n%i Training Batches Created...\n" % trainBatchCount)
        for batch in range(valiBatchCount):
            valiBatches[batch] = valiData[(batch * self.batchSize): ((batch + 1) * self.batchSize), 0:7]
        print("\n%i Validation Batches Created...\n" % valiBatchCount)

        return [self.trainBatches, valiBatches]

class ActivationFunctions():
    def __init__(self):
        pass

    def relu(self, X):
        pass

    def derivRelu(self, X):
        return X

    def leakyRelu(self, X):
        return X

    def derivLeakyRelu(self, X):
        return X

class NeuralNetwork():
    def __init__(self):
        self.actFunct = ActivationFunctions()
        self.data = DatasetPreperation()
        # General
        self.learning_rate = 0.01  # could be cool and make this variable
        # Network Architecture
        self.layerShapes = [1, 2, 2, 2, 2]  # relative sizes of input & hidden layers NOT INCLUDING OUTPUT

    def createNetwork(self, data):
        shapes = self.layerShapes
        layerCount = len(self.layerShapes)  # not including output
        # Set up the architecture
        inputCount = 6
        for layer in range(layerCount):
            shapes[layer] = shapes[layer] * inputCount
        # Initialize all layers
        self.M = [ndarray] * layerCount
        self.Z = [ndarray] * layerCount
        self.A = [ndarray] * layerCount
        # Add the weight & bias layers
        self.W = [ndarray] * layerCount
        self.B = [ndarray] * layerCount

        shapes.append(1)  # add the output neuron
        for layer in range(layerCount):
            self.W[layer] = np.zeros_like(np.random.rand(shapes[layer], shapes[layer + 1]))
            self.B[layer] = np.ones_like(np.random.rand(1, shapes[layer + 1]))
            print(self.B[layer].shape)

        print("Network Built...")

    def pickTrainBatch(self, batches):
        controller = MainOperation()
        trainBatches = batches[0]
        batchCount = len(trainBatches)
        pick = controller.currentBatch
        return trainBatches[pick]

    def pickValiBatch(self, batches):
        valiBatches = batches[1]
        batchCount = len(valiBatches)
        pick = random.randint(0, (batchCount - 1))
        return valiBatches[pick]

    def feedForward(self, batch):
        batchX = batch[:, 0:6]
        batchY = batch[:, [6]]
        for layer in range(len(self.layerShapes) - 1):
            if layer == 0:
                self.M[layer] = np.matmul(batchX, self.W[layer])
                self.Z[layer] = np.add(self.M[layer], self.B[layer])
                self.A[layer] = self.actFunct.leakyRelu(self.Z[layer])
            else:
                self.M[layer] = np.matmul(self.A[layer - 1], self.W[layer])
                self.Z[layer] = np.add(self.M[layer], self.B[layer])
                self.A[layer] = self.actFunct.leakyRelu(self.Z[layer])

        yhat = self.A[(len(self.layerShapes) - 2)]
        return batchY, yhat

    def backProp(self, Y, Yhat):

        lossBatch = (1 / Y.shape[0]) * (np.subtract(Y, Yhat)) ** 2

        dL_dYhatBatch = -1 * 1 / (Y.shape[0] / 2) * (np.subtract(Y, Yhat))  # double check this

        # initialize gradients
        layers = 5
        batchSize = self.data.batchSize
        for sample in range(batchSize):
            dL_dYhat = dL_dYhatBatch[sample]

            dA_dZ = [ndarray] * layers
            dZ_dM = [ndarray] * layers
            dM_dA = [ndarray] * layers

            dL_dA = [ndarray] * layers
            dL_dZ = [ndarray] * layers
            dL_dM = [ndarray] * layers

                # Simple layer to layer
            for layer in reversed(range(layers)):
                dA_dZ[layer] = np.transpose(self.actFunct.derivLeakyRelu(self.Z[layer]))
                dZ_dM[layer] = np.ones_like(self.B[layer])
                dM_dA[layer] = self.W[layer]  # probably need to reshape
                # Loss associated with each layer
            for layer in reversed(range(layers)):
                print(layer)
                if layer == (layers - 1):
                    dL_dA[layer] = dL_dYhat
                else:
                    print(dL_dM[layer+1])
                    print(dM_dA[layer+1])
                    dL_dA[layer] = np.dot(dL_dM[layer+1], dM_dA[layer+1])
                dL_dZ[layer] = np.dot(dL_dA[layer], dA_dZ[layer])
                dL_dM[layer] = np.dot(dL_dZ[layer], dZ_dM[layer])
        return lossBatch


controller = MainOperation()
controller.main(controller.dataset)