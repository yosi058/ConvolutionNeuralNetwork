import os
import numpy
import pandas as pd
import numpy as np
import random as random
from math import sqrt
from scipy import signal

LR = 0.01
LOAD = True
dirctory = "C:\\Users\\yosef\\PycharmProjects\\testCNN"


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
            np.random.seed(42)
            self.weights = sqrt(2 / ((
                                             size_matrix_curr_layer * size_matrix_curr_layer) * 9 * num_features_current_layer)) * np.random.randn(
                num_weights_next_layer, num_features_current_layer, 3, 3)
            # self.weights = np.random.uniform(low=-0.01, high=0.01,
            #                                 size=(num_weights_next_layer, num_features_current_layer, 3, 3))
            self.delta_weights = np.zeros(self.weights.shape)

    def convelotion(self, next_layer):
        next_layer.features = np.zeros((next_layer.num_features, next_layer.size_matrix, next_layer.size_matrix))
        for feature_map in range(next_layer.num_features):  # 3x32x32 - >16x32x32
            for current_feature_map in range(self.num_features):
                next_layer.features[feature_map] += signal.correlate2d(self.features[current_feature_map],
                                                                       # 16-> 3x(3x3)
                                                                       self.weights[feature_map][current_feature_map],
                                                                       mode='same')  # TODO.sum(axis=0)
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
                new_feature[i][j] = feature_map[i * 2:i * 2 + 2, j * 2:j * 2 + 2].max()
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
        for feature_map in range(down_fill_layer.num_features):  # 16
            for current_feature_map in range(self.num_features):  # 32
                temp_feature = signal.correlate2d(self.features[current_feature_map],
                                                  original_layer.weights[current_feature_map][feature_map],
                                                  mode='same')  # TODO.sum(axis=0)
                down_fill_layer.features[feature_map] += (original_layer.features[feature_map] > 0) * \
                                                         temp_feature

    def update_weights(self, next_layer_error):
        activate_feature = np.maximum(self.features, 0)
        activate_feature = np.minimum(activate_feature, 1)
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
            error_weights = sum_errors_of_each_feature_map.transpose(1, 0).reshape((activate_feature.shape[0], 3, 3))
            self.delta_weights[one_weight_layer] += error_weights

            # self.weights[one_weight_layer] = self.weights[one_weight_layer] + (LR * delta_weights)


# Do run noise
NOISE = False
UPDATE = False


class NeuralNetwork(object):
    def __init__(self):
        # np.random.seed(42)
        # parameters
        self.inputSize = 2048
        self.outputSize = 10
        self.hiddenSize_one = 300
        # Do load the weights from files
        # Adding one to the bias
        # self.W1 = np.random.uniform(low=-0.01, high=0.01,
        #                            size=(self.inputSize + 1, self.hiddenSize_one + 1))  # 1025x301
        # np.random.seed(42)
        self.W1 = sqrt(2 / (self.inputSize + 1 + self.hiddenSize_one + 1)) * np.random.randn(self.inputSize + 1,
                                                                                             self.hiddenSize_one + 1)
        # np.random.seed(42)
        self.W2 = sqrt(2 / (self.hiddenSize_one + 1 + self.outputSize)) * np.random.randn(self.hiddenSize_one + 1,
                                                                                          self.outputSize)
        # self.W2 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenSize_one + 1, self.outputSize))  # 301x10
        self.dalta_W1 = np.zeros(self.W1.shape)
        self.dalta_W2 = np.zeros(self.W2.shape)

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
        # self.output=self.temp_output
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
                    # temp.append(min(1, j))
                    temp.append(j)

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
        self.hidden_error[0][-1] = 0  # change bias to 0
        # func(w_ij*error_j)
        hidden_input_deriv = self.activationFunction(first_vector[0], deriv=True)
        self.input_mult_error = np.dot(self.hidden_error, self.W1.T)  # 1x301 X 301x1025 -> 1x1025
        self.input_mult_error[0][-1] = 0  # change bias to 0
        self.input_error = hidden_input_deriv * self.input_mult_error  # layer one - input

        # update w
        self.dalta_W2 += self.hidden_layer_activate.T.dot(self.output_error)
        temp = self.activationFunction(first_vector[0])
        # Converting a vector to a 1 * 1 matrix
        temp = np.asmatrix(temp, dtype=float)
        temp = np.array(temp, dtype=float)
        self.dalta_W1 += temp.T.dot(self.hidden_error)
        if UPDATE:
            self.W2 += LR * self.dalta_W2
            self.W1 += LR * self.dalta_W1
            self.dalta_W2 = np.zeros(self.W2.shape)
            self.dalta_W1 = np.zeros(self.W1.shape)

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
        self.layer_tow_max_pool = Layer(32, 8, -1, False, False)
        # self.layer_three = Layer(64, 8, -1, False, False)
        # self.layer_three_max_pool = Layer(64, 4, -1, False, False)
        self.NN = NeuralNetwork()
        self.counter = 0
        if LOAD:
            shape = np.loadtxt(os.path.join("w_input_layer.csv"), delimiter=',')
            self.layer_input.weights = shape.reshape(16, 3, 3, 3)
            shape = np.loadtxt(os.path.join("w_layer_one_max_pool.csv"), delimiter=',')
            self.layer_one_max_pool.weights = shape.reshape(32, 16, 3, 3)
            self.NN.W1 = np.loadtxt(os.path.join("w1.csv"), delimiter=',')
            self.NN.W2 = np.loadtxt(os.path.join("w2.csv"), delimiter=',')

        # self.save_error_layer_three = Layer(64, 8, -1, False, False)
        # self.save_error_layer_two = Layer(32, 16, -1, False, False)
        # self.save_error_layer_one = Layer(16, 32, -1, False, False)

    def create_layer_input(self, line):
        new_feature_map = np.array(
            [line[0:1024].reshape(32, 32), line[1024:2048].reshape(32, 32), line[2048:].reshape(32, 32)])
        return new_feature_map

    def normalize(self, feature_map):
        for j in range(len(feature_map)):
            if feature_map[j].std() != 0:
                feature_map[j] = (feature_map[j] - feature_map[j].mean()) / feature_map[j].std()
            else:
                feature_map[j] = 0
        return feature_map

    def feedForfoward(self, line):
        line = self.create_layer_input(line)
        # line = self.normalize(line)
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
        flat = np.asmatrix(flat, dtype=float)
        flat = np.array(flat, dtype=float)
        self.NN.backward(flat, correct_output, output_forward)  # back in full connected
        error_input = self.NN.reverse_flatten()
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

        # self.save_error_layer_one.features += one_layer_error.features
        # self.save_error_layer_two.features += tow_layer_error.features

        ######################## update w
        # self.layer_tow_max_pool.update_weights(three_layer_error)  # update first - layer weight

        self.layer_one_max_pool.update_weights(tow_layer_error)
        self.layer_input.update_weights(one_layer_error)
        if UPDATE:
            self.layer_one_max_pool.weights += self.layer_one_max_pool.delta_weights * LR
            self.layer_input.weights += self.layer_input.delta_weights * LR
            self.layer_one_max_pool.delta_weights = np.zeros(self.layer_one_max_pool.delta_weights.shape)
            self.layer_input.delta_weights = np.zeros(self.layer_input.delta_weights.shape)

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

validate_data = pd.read_csv("test.csv", header=None)
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
if not LOAD:
    for k in range(100):
        print("the number of epoch is" + str(k))
        if k == 7:
            LR = 0.005
        if k == 11:
            LR = 0.001
        if k == 15:
            LR = 0.0004
        counter_train = 0
        my_model.counter = 0

        for i in range(16000):
            if i % 22 == 0 and i > 0:
                UPDATE = True
            else:
                UPDATE = False
            if i % 1000 == 0:
                print("pass " + str(i) + " lines")
            current_output = np.zeros(10)
            # The right result
            current_output[int(result_line[i]) - 1] = 1
            # Create a vector of the contemporary line
            current_output = np.array(current_output, dtype=float)
            cur_row = train_data[i]
            cur_row = np.array(cur_row, dtype=float)
            my_model.train(cur_row, current_output)
        # if k >= 0:
        #     path = str(k)
        #     if not os.path.exists(str(path)):
        #         os.mkdir(path)
        #     new_dir =dirctory+ "\\" + path
        #     shape = my_model.layer_input.weights.reshape(432, 1)
        #     np.savetxt(os.path.join(new_dir, 'w_input_layer.csv'), shape, delimiter=",", fmt='%f')
        #     shape = my_model.layer_one_max_pool.weights.reshape(4608, 1)
        #     np.savetxt(os.path.join(new_dir, 'w_layer_one_max_pool.csv'), shape, delimiter=",", fmt='%f')
        #     np.savetxt(os.path.join(new_dir, 'w1.csv'), my_model.NN.W1, delimiter=",", fmt='%f')
        #     np.savetxt(os.path.join(new_dir, 'w2.csv'), my_model.NN.W2, delimiter=",", fmt='%f')

        print("Has the success of train is", (my_model.counter / 16000) * 100)
current_output_validate = []
output = []
counter = 0
not_know = 0
for t in range(1000):
    cur_row = validate_data[t]
    cur_row = np.array(cur_row, dtype=float)
    output_forward = my_model.feedForfoward(cur_row)
    output_forward = np.array(output_forward, dtype=float)
    # Finding the Index of the Largest Organ
    max_index = np.argmax(output_forward, axis=0)
    max_index = int(max_index)
    output.append(max_index + 1)

output = np.array(output, dtype=int)
# numpy.savetxt('output.txt', output, newline="\n")
numpy.savetxt(fname='output.txt', X=output.astype(int), fmt='%.0f')
