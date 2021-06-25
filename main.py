import numpy
import pandas as pd
import numpy as np
import random as random
from math import sqrt
import time
import os
from scipy import signal

LR = 0.001

np.random.seed(30)


# 3x32x32 , 16x3x3x3
class Layer:
    def __init__(self, num_features_current_layer, size_matrix_curr_layer, num_weights_next_layer, features_map=None,
                 create_weight=True):
        self.num_features = num_features_current_layer  # or list of w []
        self.size_matrix = size_matrix_curr_layer
        if features_map is False:
            self.features = np.zeros(
                (num_features_current_layer, size_matrix_curr_layer, size_matrix_curr_layer))  # init 16*32*32
        else:
            self.features = features_map
        if create_weight:
            # self.weights = sqrt(2/size_matrix_curr_layer*9*num_features_current_layer)*np.random.randn(num_weights_next_layer,num_features_current_layer,3,3)
            self.weights = np.random.uniform(low=-0.01, high=0.01,
                                             size=(num_weights_next_layer, num_features_current_layer, 3, 3))

    def convelotion(self, next_layer):
        next_layer.features = np.zeros((next_layer.num_features, next_layer.size_matrix, next_layer.size_matrix))
        for feature_map in range(next_layer.num_features):  # 3x32x32 - >16x32x32
            for current_feature_map in range(self.num_features):
                next_layer.features[feature_map] += signal.correlate2d(self.features[current_feature_map],
                                                                       self.weights[feature_map][current_feature_map],
                                                                       mode='same').sum(axis=0)
        next_layer.features = np.maximum(next_layer.features, 0)
        next_layer.features = np.minimum(next_layer.features, 1)
        # function activate RELU

    # return next_layer

    def maxpool(self, next_layer):
        for feature_map in range(next_layer.num_features):
            next_layer.features[feature_map] = self.get_max_pool(self.features[feature_map],
                                                                 next_layer.size_matrix)  # get feature by max-pool

    # return next_layer

    def get_max_pool(self, feature_map, size_feature):
        new_feature = np.empty((size_feature, size_feature))  # new size - half from the original
        for i in range(size_feature):
            for j in range(size_feature):
                new_feature[i][j] = max(feature_map[2 * i, 2 * j], feature_map[2 * i, 2 * j + 1],
                                        feature_map[2 * i + 1, 2 * j], feature_map[2 * i + 1, 2 * j + 1])
        return new_feature  # fill by yhe max from each 4 numbers.

    def reverse_max_pool(self, prv_layer, original_layer):
        for feature_map in range(original_layer.num_features):
            prv_layer.features[feature_map] = self.get_reverse_max_pool(original_layer.features[feature_map],
                                                                        prv_layer.size_matrix, self.features[
                                                                            feature_map])  # get feature by max-pool_reverse

    def get_reverse_max_pool(self, feature_map, size_matrix, error_layer_feature):
        new_feature = np.zeros((size_matrix, size_matrix))
        size = int(size_matrix / 2)
        for i in range(size):
            for j in range(size):
                max_val = max(feature_map[2 * i, 2 * j], feature_map[2 * i, 2 * j + 1], feature_map[2 * i + 1, 2 * j],
                              feature_map[2 * i + 1, 2 * j + 1])
                if max_val == feature_map[2 * i, 2 * j]:
                    max_i = 2 * i
                    max_j = 2 * j
                elif max_val == feature_map[2 * i, 2 * j + 1]:
                    max_i = 2 * i
                    max_j = 2 * j + 1
                elif max_val == feature_map[2 * i + 1, 2 * j]:
                    max_i = 2 * i + 1
                    max_j = 2 * j
                else:
                    max_i = 2 * i + 1
                    max_j = 2 * j + 1
                new_feature[max_i][max_j] = error_layer_feature[i][j]
        return new_feature

    def flatten_layer(self):
        co_layer = self.features
        co_layer = np.asmatrix(co_layer.flatten())
        return np.array(co_layer)

    def get_reverse_conv(self, down_fill_layer, original_layer):
        # TODO - get over the corolate line and find mistakes.
        # new_w = original_layer.weights.transpose(1,0,2,3)
        for feature_map in range(down_fill_layer.num_features):
            for current_feature_map in range(self.num_features):
                temp_feature = signal.correlate2d(self.features[current_feature_map],
                                                  original_layer.weights[current_feature_map][feature_map],
                                                  mode='same').sum(axis=0)
                down_fill_layer.features[feature_map] += (original_layer.features[feature_map] > 0) * \
                                                         temp_feature

    def update_weights(self, next_layer_error):
        activate_feature = np.maximum(self.features, 0)
        activate_feature = np.minimum(self.features, 1)
        # activate function Relu
        for one_weight_layer in range(
                next_layer_error.num_features):  # one dimenssion from 4 of the weight matrix layer
            last_layer_error_according_weights = np.zeros((9, self.size_matrix, self.size_matrix))
            last_layer_error_according_weights[0][:-1, :-1] = next_layer_error.features[one_weight_layer][1:, 1:]
            last_layer_error_according_weights[1][:-1, :] = next_layer_error.features[one_weight_layer][1:, :]
            last_layer_error_according_weights[2][:-1, 1:] = next_layer_error.features[one_weight_layer][1:, :-1]
            last_layer_error_according_weights[3][:, :-1] = next_layer_error.features[one_weight_layer][:, 1:]
            last_layer_error_according_weights[4] = next_layer_error.features[one_weight_layer]
            last_layer_error_according_weights[5][:, 1:] = next_layer_error.features[one_weight_layer][:, :-1]
            last_layer_error_according_weights[6][1:, :-1] = next_layer_error.features[one_weight_layer][:-1, 1:]
            last_layer_error_according_weights[7][1:, :] = next_layer_error.features[one_weight_layer][:-1, :]
            last_layer_error_according_weights[8][1:, 1:] = next_layer_error.features[one_weight_layer][:-1, :-1]
            multiple_errors_by_previous_layer = activate_feature[None, :, :, :] * last_layer_error_according_weights[:,
                                                                                  None, :, :]
            sum_errors_of_each_feature_map = multiple_errors_by_previous_layer.sum(axis=(2, 3))
            delta_weights = sum_errors_of_each_feature_map.transpose(1, 0).reshape((activate_feature.shape[0], 3, 3))
            self.weights[one_weight_layer] = self.weights[one_weight_layer] + (LR * delta_weights)


# Do run noise
NOISE = False
LOAD = True


class NeuralNetwork(object):
    def __init__(self):
        # np.random.seed(42)
        # parameters
        self.inputSize = 2048
        self.outputSize = 10
        self.hiddenSize_one = 600
        # Do load the weights from files
        # Adding one to the bias
        self.W1 = np.random.uniform(low=-0.01, high=0.01,
                                    size=(self.inputSize + 1, self.hiddenSize_one + 1))  # 1025x301
        self.W2 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenSize_one + 1, self.outputSize))  # 301x10

    # Receiving a vector and returning the output
    def feedForward(self, first_vector):
        # forward propogation through the network
        first_vector = np.append(first_vector, -1)
        # the bias
        # dot the input with the first matrix
        self.hidden_layer = np.dot(first_vector, self.W1)  # 1x301
        # Activation of the activation function
        self.hidden_layer_activate = self.activationFunction(self.hidden_layer)
        # the bias
        self.hidden_layer_activate[self.hiddenSize_one] = -1
        # Converting a vector to a 1 * 1 matrix
        self.hidden_layer_activate = np.asmatrix(self.hidden_layer_activate, dtype=float)
        self.hidden_layer_activate = np.array(self.hidden_layer_activate, dtype=float)

        # dot the hidden layer 2 and second set of weights
        self.temp_output = np.dot(self.hidden_layer_activate, self.W2)
        self.output = self.activationFunction(self.temp_output[0])
        return self.output

    # Activation function
    def activationFunction(self, s, deriv=False):
        count = 0
        temp = []
        if (deriv == True):
            for i in s:
                if i > 0:
                    temp.append(1)
                # s[count] = 1
                else:
                    temp.append(0)
                # s[count] = 0
                count += 1
        else:
            for j in s:
                if j < 0:
                    temp.append(0)
                # s[count] = 0
                else:
                    temp.append(min(1, j))
                    # s[count] = min(1, j)
                count += 1
        return np.array(temp)

    # backward propogate through the network
    def backward(self, first_vector, correct_output, output_forward):
        # the bias
        first_vector = np.append(first_vector, 0)
        # Converting a vector to a 1 * 1 matrix
        first_vector = np.asmatrix(first_vector, dtype=float)
        first_vector = np.array(first_vector, dtype=float)
        # Converting a vector to a 1 * 1 matrix
        correct_output = np.asmatrix(correct_output, dtype=float)
        correct_output = np.array(correct_output, dtype=float)
        # Converting a vector to a 1 * 1 matrix
        output_forward = np.asmatrix(output_forward, dtype=float)
        output_forward = np.array(output_forward, dtype=float)

        # error in output 1x10
        self.output_error = correct_output - output_forward
        # sigma(w_ij*error_j)
        self.hidden_mult_error = np.dot(self.output_error, self.W2.T)  # 1x10 x10x301 -> 1x301
        # Activation of the activation function
        self.hidden_layer = np.asmatrix(self.hidden_layer, dtype=float)
        self.hidden_layer = np.array(self.hidden_layer, dtype=float)
        hidden_layer_deriv = self.activationFunction(self.hidden_layer[0], deriv=True)
        # f'(x_i)* func(w_ij*error_j)
        self.hidden_error = hidden_layer_deriv * self.hidden_mult_error
        # func(w_ij*error_j)
        hidden_input_deriv = self.activationFunction(first_vector[0], deriv=True)
        self.input_mult_error = np.dot(self.hidden_error, self.W1.T)  # 1x301 X 301x1025 -> 1x1025
        self.input_error = hidden_input_deriv * self.input_mult_error  # layer one - input
        # update w
        self.W2 += LR * self.hidden_layer_activate.T.dot(self.output_error)
        temp = self.activationFunction(first_vector[0])
        # Converting a vector to a 1 * 1 matrix
        temp = np.asmatrix(temp, dtype=float)
        temp = np.array(temp, dtype=float)
        self.W1 += LR * temp.T.dot(self.hidden_error)

    # The function randomly resets 10% of the input
    def noise(self, cur_row):
        # 10%
        range_row = (self.inputSize) * 0.1
        # convert to int
        range_row = int(range_row)
        for p in range(range_row):
            # choose index
            index = np.random.randint(0, self.inputSize - 1)
            cur_row[index] = 0
        return cur_row

    # The function activates the neural network and then learns the error
    def train(self, cur_row, correct_output):
        # Do turn on Noise
        if NOISE:
            cur_row = self.noise(cur_row)
        output_forward = self.feedForward(cur_row)
        self.backward(cur_row, correct_output, output_forward)

    def reverse_flatten(self):
        self.input_error = np.array(self.input_error, dtype=float)
        self.input_error = np.delete(self.input_error, self.input_error.size - 1)
        return self.input_error.reshape(32, 8, 8)
    # return self.input_error.reshape(64, 4, 4)


class Model:
    def __init__(self):
        self.layer_input = Layer(3, 32, 16, False, True)  # create input layer
        self.layer_one = Layer(16, 32, -1, False, False)  # create first kayer without max-pool
        self.layer_one_max_pool = Layer(16, 16, 32, False, True)  # create max-pool first layer
        self.layer_tow = Layer(32, 16, -1, False, False)
        self.layer_tow_max_pool = Layer(32, 8, 64, False, True)
        self.layer_three = Layer(64, 8, -1, False, False)
        self.layer_three_max_pool = Layer(64, 4, -1, False, False)
        self.NN = NeuralNetwork()
        self.counter = 0

    def create_layer_input(self, line):
        new_feature_map = np.array(
            [line[0:1024].reshape(32, 32), line[1024:2048].reshape(32, 32), line[2048:].reshape(32, 32)])
        return new_feature_map

    def feedForfoward(self, line):
        line = self.create_layer_input(line)
        self.layer_input.features = line
        self.layer_input.convelotion(self.layer_one)
        self.layer_one.maxpool(self.layer_one_max_pool)
        self.layer_one_max_pool.convelotion(self.layer_tow)
        self.layer_tow.maxpool(self.layer_tow_max_pool)
        # self.layer_tow_max_pool.convelotion(self.layer_three)
        # self.layer_three.maxpool(self.layer_three_max_pool)
        # self.flat = self.layer_three_max_pool.flatten_layer()
        # self.flat = self.layer_tow_max_pool.flatten_layer()
        self.flat = self.layer_tow_max_pool.flatten_layer()
        self.output = self.NN.feedForward(self.flat)
        return self.output

    def backFoward(self, correct_output, output_forward):
        flat = self.flat
        # flat = self.flat
        flat = np.asmatrix(flat, dtype=float)
        flat = np.array(flat, dtype=float)
        self.NN.backward(flat, correct_output, output_forward)  # back in full connected
        error_input = self.NN.reverse_flatten()  # 64x4x4
        # three_layer_error_max_pool = Layer(64, 4, -1, error_input, False)  # first layer back
        # three_layer_error = Layer(64, 8, -1, False, False)
        # three_layer_error_max_pool.reverse_max_pool(three_layer_error, self.layer_three)  # 64x8x8
        # tow_layer_max_pool_error = Layer(32, 8, -1, False, False)
        tow_layer_max_pool_error = Layer(32, 8, -1, error_input, False)
        # three_layer_error.get_reverse_conv(tow_layer_max_pool_error,
        #                                   self.layer_tow_max_pool)  # tow_layer is the original
        tow_layer_error = Layer(32, 16, -1, False, False)
        tow_layer_max_pool_error.reverse_max_pool(tow_layer_error, self.layer_tow)
        one_layer_max_pool_error = Layer(16, 16, -1, False, False)
        tow_layer_error.get_reverse_conv(one_layer_max_pool_error, self.layer_one_max_pool)
        one_layer_error = Layer(16, 32, -1, False, False)
        one_layer_max_pool_error.reverse_max_pool(one_layer_error, self.layer_one)

        ######################## update w
        # self.layer_tow_max_pool.update_weights(three_layer_error)  # update first - layer weight
        self.layer_one_max_pool.update_weights(tow_layer_error)
        self.layer_input.update_weights(one_layer_error)

    def train(self, cur_row, correct_output):
        self.feedForfoward(cur_row)
        max_index = np.argmax(self.output, axis=0)
        correct_index = np.argmax(correct_output, axis=0)
        # print("output " + str(max_index) + "  expected out is " + str(correct_index))

        # The amount of success
        if correct_output[max_index] == 1:
            self.counter += 1
        self.backFoward(correct_output, self.output)


# Upload the train file
train_data = pd.read_csv("train.csv", header=None)
# The first column with the answers
result_line = train_data.loc[:, 0]
# Remove the first column
train_data = train_data.drop(columns=0)
# Upload the validate file

validate_data = pd.read_csv("validate.csv", header=None)
# The first column with the answers
result_line_validate = validate_data.loc[:, 0]
# Remove the first column
validate_data = validate_data.drop(columns=0)

# An array that will hold the right results
current_output = np.zeros(10)
# Create a neural network (if true the weights would be initialized from files)
my_model = Model()
train_data = train_data.rename(columns=lambda c: c - 1).to_numpy()
validate_data = validate_data.rename(columns=lambda c: c - 1).to_numpy()
for k in range(100):
    if k == 14:
        LR = 0.8 * LR
    print("the number of epoch is" + str(k))
    counter_train = 0
    my_model.counter = 0
    for i in range(8000):
        if i % 100 == 0:
            print("pass " + str(i) + " lines")
        current_output = np.zeros(10)
        # The right result
        current_output[result_line[i] - 1] = 1
        # Create a vector of the contemporary line
        current_output = np.array(current_output, dtype=float)
        cur_row = train_data[i]
        cur_row = np.array(cur_row, dtype=float)
        my_model.train(cur_row, current_output)

    print("Has the success of train is", (my_model.counter / 8000) * 100)
    current_output_validate = []
    counter = 0
    for t in range(1000):
        current_output_validate = np.zeros(10)
        # The right result
        current_output_validate[result_line_validate[t] - 1] = 1
        # Create a vector of the contemporary line
        current_output_validate = np.array(current_output_validate, dtype=float)
        # Create a vector of the contemporary line
        cur_row = validate_data[t]
        cur_row = np.array(cur_row, dtype=float)
        output_forward = my_model.feedForfoward(cur_row)
        output_forward = np.array(output_forward, dtype=float)
        # Finding the Index of the Largest Organ
        max_index = np.argmax(output_forward, axis=0)
        # The amount of success
        if current_output_validate[max_index] == 1:
            counter += 1

    percent = ((counter / 1000) * 100)

    print("Has the success of ", percent)

"""
3x32x32 -> 16x32x32 -> 16x16x16 ->32x16x16 -> 32x8x8 -> 64x8x8 -> 64x4x4 - > flat
"""
"""

class NeuralNetwork(object):
    def __init__(self, load=False):
        # parameters
        self.inputSize = 1024
        self.outputSize = 10
        self.hiddenSize_one = 200
        self.W1 = np.random.uniform(low=-0.01, high=0.01, size=(self.inputSize + 1, self.hiddenSize_one + 1))
        self.W2 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenSize_one + 1, self.outputSize))

    # self.W3[self.hiddenSize_two] = self.bias

    # weights
    # self.W1 = np.random.randn(self.inputSize, self.hiddenSize)   # (3072 x 1000) weight matrix from input to hidden layer
    # self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (1000 x 10) weight matrix from hidden to output layer

    def feedForward(self, cur_row):
        # forward propogation through the network
        # cur_row = np.array(cur_row, dtype=float)
        cur_row = np.append(cur_row, -1)
        ##
        self.hidden_layer_one = np.dot(cur_row, self.W1)  # dot the input with the first matrix
        # self.hidden_layer_one[self.hiddenSize_one] = 1
        # self.hidden_layer_one = np.append(self.hidden_layer_one, 1)
        self.hidden_error_one_sigmoid = self.sigmoid(self.hidden_layer_one)
        self.hidden_error_one_sigmoid[self.hiddenSize_one] = -1
        ########################
        self.hidden_error_one_sigmoid = np.asmatrix(self.hidden_error_one_sigmoid, dtype=float)
        self.hidden_error_one_sigmoid = np.array(self.hidden_error_one_sigmoid, dtype=float)
        # activation function sigmoid
        self.temp_output = np.dot(self.hidden_error_one_sigmoid,
                                  self.W2)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.temp_output[0])
        return output

    def sigmoid(self, s, deriv=False):
        count = 0
        if deriv:
            for i in s:
                if i > 0:
                    s[count] = 1
                else:
                    s[count] = 0
                count += 1
        else:
            for j in s:
                if j < 0:
                    s[count] = 0
                else:
                    s[count] = min(1, j)
                count += 1
        return s


    def backward(self, cur_row, correct_output, output_forward):
        # backward propogate through the network
        ###
        cur_row = np.append(cur_row, -1)
        cur_row = np.asmatrix(cur_row, dtype=float)
        cur_row = np.array(cur_row, dtype=float)
        correct_output = np.asmatrix(correct_output, dtype=float)
        correct_output = np.array(correct_output, dtype=float)
        output_forward = np.asmatrix(output_forward, dtype=float)
        output_forward = np.array(output_forward, dtype=float)
        ##

        self.output_error = correct_output - output_forward  # error in output 1x10
        self.hidden_mult_error = np.dot(self.output_error, self.W2.T)  # sigma(w_ij*error_j) 1x1000
        sigmo = self.sigmoid(self.hidden_layer_one[0], deriv=True)
        self.hidden_error = sigmo * self.hidden_mult_error  # f'(x_i)* sigma(w_ij*error_j)

        self.W3 += LS * self.hidden_layer_tow_sigmoid.T.dot(self.output_error)

        self.W2 += LS * self.hidden_error_one_sigmoid.T.dot(self.hidden_error_two)
        temp = self.sigmoid(cur_row[0])
        temp = np.asmatrix(temp, dtype=float)
        temp = np.array(temp, dtype=float)
        self.W1 += LS * temp.T.dot(self.hidden_error_one)


    def noise(self, cur_row):
        range_row = (self.inputSize) * 0.1
        range_row = int(range_row)
        for p in range(range_row):
            index = random.randint(0, self.inputSize - 1)
            cur_row[index] = 0
        return cur_row

    def train(self, cur_row, correct_output):
        cur_row = self.noise(cur_row)
        output_forward = self.feedForward(cur_row)
        self.backward(cur_row, correct_output, output_forward)
"""""
