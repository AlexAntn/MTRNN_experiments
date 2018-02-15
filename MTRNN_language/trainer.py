from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CTRNN import CTRNNModel

import time
import operator
import io
import array
from datetime import datetime

import os
import sys

def processWords(sentence, x_train, k):
    sentence = sentence.replace("\n", "").replace("\r", "")
    print(sentence)
    for f in range(0, len(sentence), 1):
        if sentence[f] == ' ' and f <= 29:
            x_train[k, f + 4,26] = 1.0
        elif sentence[f] == '.' and f <= 29:
            x_train[k, f + 4,27] = 1.0
        elif f <= 29:
            print(sentence[f])
            x_train[k, f + 4,ord(sentence[f]) - 97] = 1.0 

    return x_train 

#construction of control sequence (fixed combinations, 6 neurons, activation can be 0, 0.5 or 1.0)
def get_combination(sentence, y_train, k):
    split_sentence = sentence.replace(".", "").replace("\r", "").replace("\n", "").split(" ")
    for t in range(4, 30, 1):
        if "slide" in split_sentence and "left" in split_sentence:
                y_train[k, t, 0] = 0.5
                y_train[k, t, 1] = 0.0
                y_train[k, t, 2] = 0.0
        elif "slide" in split_sentence and "right" in split_sentence:
                y_train[k, t, 0] = 1.0
                y_train[k, t, 1] = 0.0
                y_train[k, t, 2] = 0.0
        elif "touch" in split_sentence:
                y_train[k, t, 0] = 0.0
                y_train[k, t, 1] = 0.5
                y_train[k, t, 2] = 0.0
        elif "reach" in split_sentence:
                y_train[k, t, 0] = 0.0
                y_train[k, t, 1] = 1.0
                y_train[k, t, 2] = 0.0
        elif "push" in split_sentence:
                y_train[k, t, 0] = 0.5
                y_train[k, t, 1] = 0.5
                y_train[k, t, 2] = 0.0
        elif "pull" in split_sentence:
                y_train[k, t, 0] = 1.0
                y_train[k, t, 1] = 0.5
                y_train[k, t, 2] = 0.0
        elif "point" in split_sentence:
                y_train[k, t, 0] = 1.0
                y_train[k, t, 1] = 1.0
                y_train[k, t, 2] = 0.0
        elif "grasp" in split_sentence:
                y_train[k, t, 0] = 0.0
                y_train[k, t, 1] = 0.0
                y_train[k, t, 2] = 0.5
        elif "lift" in split_sentence:
                y_train[k, t, 0] = 0.0
                y_train[k, t, 1] = 0.0
                y_train[k, t, 2] = 1.0
        if "tractor" in split_sentence:
                y_train[k, t, 3] = 0.5
                y_train[k, t, 4] = 0.0
                y_train[k, t, 5] = 0.0
        elif "hammer" in split_sentence:
                y_train[k, t, 3] = 1.0
                y_train[k, t, 4] = 0.0
                y_train[k, t, 5] = 0.0
        elif "ball" in split_sentence:
                y_train[k, t, 3] = 0.0
                y_train[k, t, 4] = 0.5
                y_train[k, t, 5] = 0.0
        elif "bus" in split_sentence:
                y_train[k, t, 3] = 0.0
                y_train[k, t, 4] = 1.0
                y_train[k, t, 5] = 0.0
        elif "modi" in split_sentence:
                y_train[k, t, 3] = 0.5
                y_train[k, t, 4] = 0.5
                y_train[k, t, 5] = 0.0
        elif "car" in split_sentence:
                y_train[k, t, 3] = 1.0
                y_train[k, t, 4] = 0.5
                y_train[k, t, 5] = 0.0
        elif "cup" in split_sentence:
                y_train[k, t, 3] = 1.0
                y_train[k, t, 4] = 1.0
                y_train[k, t, 5] = 0.0
        elif "cubes" in split_sentence:
                y_train[k, t, 3] = 0.0
                y_train[k, t, 4] = 0.0
                y_train[k, t, 5] = 0.5
        elif "spiky" in split_sentence:
                y_train[k, t, 3] = 0.0
                y_train[k, t, 4] = 0.0
                y_train[k, t, 5] = 1.0
    return y_train

######################################################################################
# This function loads data from a file, to train the network
# inputs are sequential (and always same order). 
def loadTrainingData():


    numInputNeurons = 29 # language neurons, 26 for letters plus space and stop plus "nothing"
    numControlNeurons = 3 
    stepEachSeq = 30


    sentences = ["bafffffffff", "kceeeeeeeee", "vvvvvrrrrr", "ghllllll", "uuuutttttt", "push the ball"]
    numSeq = len(sentences)
    
    # sequence of letters
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, numInputNeurons)),dtype=np.float32)

    # control sequence
    y_train = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)

    control_input = np.asarray(np.zeros((numSeq, stepEachSeq, numControlNeurons)),dtype=np.float32)
    

    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)
    
    #control_input[0, 0, :] = [0.0, 0.0, 0.0]
    control_input[0, 0, :] = [0.0, 0.0, 1.0]
    control_input[1, 0, :] = [0.0, 1.0, 0.0]
    control_input[2, 0, :] = [0.0, 1.0, 1.0]
    control_input[3, 0, :] = [1.0, 0.0, 0.0]
    control_input[4, 0, :] = [1.0, 0.0, 1.0]
    control_input[5, 0, :] = [1.0, 1.0, 0.0]

    k = 0 #number of sequences
    for sentence in sentences:
        
        for f in range(0, len(sentence), 1):
            if sentence[f] == ' ' and f <= 29:
                x_train[k, f + 4,26] = 1
                y_train[k, f+4] = 27
            elif sentence[f] == '.' and f <= 29:
                x_train[k, f + 4,27] = 1
                y_train[k, f+4] = 28
            elif f <= 29:
                x_train[k, f + 4,ord(sentence[f]) - 97] = 1
                y_train[k, f+4] = (ord(sentence[f]) - 96)
        k = k+1 

    return x_train, y_train, numSeq, stepEachSeq, control_input


def plot(loss_list):
    plt.subplot(2,3,1)
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)


x_train, y_train, numSeq, stepEachSeq, control_input = loadTrainingData()


print("data loaded")

lang_input = 29 # I/O layer
lang_dim1 = 100 # fast context
lang_dim2 = 30 # slow context (without control neurons)
control_dim = 3 # control neurons

LEARNING_RATE = 5 * 1e-3

NEPOCH = 2500 # number of times to train each sentence
    
my_path= os.getcwd()
figure_path = os.path.join(my_path, "matrix/")


MTRNN = CTRNNModel([lang_input, lang_dim1, lang_dim2, control_dim], [2, 5, 60, 60], stepEachSeq, control_dim, lang_input, LEARNING_RATE)


#plt.ion()
#plt.figure()
#plt.show()
loss_list = []

threshold = 0.025

MTRNN.sess.run(tf.global_variables_initializer())


init_state_IO = np.zeros([numSeq, lang_input], dtype = np.float32)
init_state_fc = np.zeros([numSeq, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeq, lang_dim2], dtype = np.float32)
init_state_cn = np.zeros([numSeq, control_dim], dtype = np.float32)

best_loss = 1000.0
epoch_idx = 0
while best_loss > threshold:
    print("Training epoch " + str(epoch_idx))
    _total_loss, _train_op, _state_tuple, _softmax, _logits = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple, MTRNN.softmax, MTRNN.logits], feed_dict={MTRNN.x:control_input, MTRNN.y:y_train, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc, 'initU_3:0':init_state_cn, 'initC_3:0':init_state_cn})
    loss_list.append(_total_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(_total_loss))
    plot(loss_list)
    if _total_loss < best_loss: # only save when loss is lower
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(_total_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss = _total_loss
    epoch_idx += 1
    if epoch_idx > NEPOCH:
        break

# TEST #

print("testing")

init_state_IO = np.zeros([1, lang_input], dtype = np.float32)
init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)
init_state_cn = np.zeros([1, control_dim], dtype = np.float32)

new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
new_input = np.asarray(np.zeros((1, stepEachSeq, control_dim)),dtype=np.float32)
new_input[0, :, :] = control_input[1, :, :]
new_output[0, :] = y_train[1, :]
print(new_output)
print(new_input)

_total_loss, _state_tuple, _softmax, _logits, _labels_y = MTRNN.sess.run([MTRNN.total_loss, MTRNN.state_tuple, MTRNN.softmax, MTRNN.logits, MTRNN.labels_y], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc, 'initU_3:0':init_state_cn, 'initC_3:0':init_state_cn})

print(_labels_y)
#print(_logits)
sentence = ""
for i in range(1*30):
    if i%30 == 0:
        print(sentence)
        print("########################")
        sentence = ""
    for g in range(28):
        if _softmax[i,g] == max(_softmax[i]): #and max(_softmax[i]) > 0.6:
            if g <27:
                sentence += chr(96 + g)
            if g == 27:
                sentence += " "
            if g == 28:
                sentence += "."
print(sentence)


print("save path for model: ",save_path)

#MTRNNTest = CTRNNModelTest([lang_input, lang_dim1, lang_dim2, control_dim], [2, 5, 60, 60], stepEachSeq, control_dim, lang_input, LEARNING_RATE, numSeq)

#tf.reset_default_graph()
#MTRNNTest.saver.restore(MTRNNTest.sess, save_path)

#for i in range(numSeq):

#new_output_temp = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)
#new_input_temp = np.asarray(np.zeros((numSeq, stepEachSeq, control_dim)),dtype=np.float32)
#new_output_temp[1, :] = y_train[4, :]
#new_input_temp[1, :, :] = control_input[4, :, :]

#_total_loss, _state_tuple, _softmax, _logits, _labels_y = MTRNNTest.sess.run([MTRNNTest.total_loss, MTRNNTest.state_tuple, MTRNNTest.softmax, MTRNNTest.logits, MTRNNTest.labels_y], feed_dict={MTRNNTest.x:new_input_temp, MTRNNTest.y:new_output_temp})
#
#print(_labels_y)
#print(_logits)
#sentence = ""
#for i in range(numSeq*30):
#    if i%30 == 0:
#        print(sentence)
#        print("########################")
#        sentence = ""
#    for g in range(29):
#        if _softmax[i,g] == max(_softmax[i]): #and max(_softmax[i]) > 0.6:
#            if g <27:
#                sentence += chr(96 + g)
#            if g == 27:
#                sentence += " "
#            if g == 28:
#                sentence += "."
#print(sentence)

#plt.ioff()
#plt.show()
MTRNN.sess.close()
#MTRNNTest.sess.close()

