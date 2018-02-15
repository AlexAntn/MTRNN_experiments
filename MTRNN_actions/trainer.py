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

import itertools

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

def process_lang_data(verb, obj):
    if verb >= 0.0 and verb < 0.1:
        sentence = "slide left"
    elif verb >= 0.1 and verb < 0.2:
        sentence = "slide right"
    elif verb >= 0.2 and verb < 0.3:
        sentence = "touch"
    elif verb >= 0.3 and verb < 0.4:
        sentence = "reach"
    elif verb >= 0.4 and verb < 0.5:
        sentence = "push"
    elif verb >= 0.5 and verb < 0.6:
        sentence = "pull"
    elif verb >= 0.6 and verb < 0.7:
        sentence = "point"
    elif verb >= 0.7 and verb < 0.8:
        sentence = "grasp"
    else:
        sentence = "lift"
    if obj >= 0.0 and obj < 0.1:
        sentence = sentence + " the " + "tractor"
    elif obj >= 0.1 and obj < 0.2:
        sentence = sentence + " the " + "hammer"
    elif obj >= 0.2 and obj < 0.3:
        sentence = sentence + " the " + "ball"
    elif obj >= 0.3 and obj < 0.4:
        sentence = sentence + " the " + "bus"
    elif obj >= 0.4 and obj < 0.5:
        sentence = sentence + " the " + "modi"
    elif obj >= 0.5 and obj < 0.6:
        sentence = sentence + " the " + "car"
    elif obj >= 0.6 and obj < 0.7:
        sentence = sentence + " the " + "cup"
    elif obj >= 0.7 and obj < 0.8:
        sentence = sentence + " the " + "cubes"
    else:
        sentence = sentence + " the " + "spiky"
    sentence = sentence + "."

    return sentence

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

def get_binary_combination(control_input, actions, objects, combinations, numSeq):
    sentences = []
    k = 0
    for i in range(len(actions)):
        for t in range(len(objects)):
            sentences += [actions[i] + " the " + objects[t] + "."]
            control_input[k, 0, 0:8] = list(combinations[i+1]) + list(combinations[t+1])
            #print(k)
            #print(control_input[i, 0, 0:8])
            #print(sentences[-1])
            k += 1
    #print(sentences)
    raw_input()
    
    return control_input, sentences

def get_sinwave(stepsEachSeq, freq, phi):
    sample = stepsEachSeq
    x = np.arange(sample)
    y = np.sin((2 * np.pi * freq * x + phi*stepsEachSeq)/ stepsEachSeq)
    return y

def loadTrainingData():

    output_neurons = 41
    numInputNeurons = 100 # language neurons, 26 for letters plus space and stop plus "nothing"
    numControlNeurons = 45 
    stepEachSeq = 100

    combinations = ["".join(seq) for seq in itertools.product("01", repeat = 2)]

    numSeq = 10 #from dataset
    #numSeq = len(actions) * len(objects)
    
    # sequence of vectors of encoders, to compare (x_train[0] corresponds to init_state)
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, numInputNeurons)),dtype=np.float32)

    # this is used for the softmax - will have to be changed?
    y_train = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)

    # control sequence working as input
    control_input = np.asarray(np.zeros((numSeq, stepEachSeq, numControlNeurons)),dtype=np.float32)

    #different possible control sequences
    control_sequences = np.asarray(np.zeros((81, stepEachSeq, numControlNeurons)),dtype=np.float32)
    

    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)
    

    control_input[0, :, 0:4] = [0.0, 0.0, 0.0, 1.0]
    control_input[1, :, 0:4] = [0.0, 0.0, 1.0, 0.0]
    control_input[2, :, 0:4] = [0.0, 0.0, 1.0, 1.0]
    control_input[3, :, 0:4] = [0.0, 1.0, 0.0, 0.0]
    control_input[4, :, 0:4] = [0.0, 1.0, 0.0, 1.0]
    control_input[5, :, 0:4] = [0.0, 1.0, 1.0, 0.0]
    control_input[6, :, 0:4] = [0.0, 1.0, 1.0, 1.0]
    control_input[7, :, 0:4] = [1.0, 0.0, 0.0, 0.0]
    control_input[8, :, 0:4] = [1.0, 0.0, 0.0, 1.0]
    control_input[9, :, 0:4] = [1.0, 0.0, 1.0, 0.0]

    dataFile = open("mtrnnTD.txt", 'r')
    
    k = 0 #number of sequences
    while True:
        line = dataFile.readline()
        if line == "":
            break
        if line.find("SEQUENCE") != -1:
            #print "found sequence"
            for i in range(0, stepEachSeq):
                line = dataFile.readline()
                line_data = line.split("\t")
                line_data[-1] = line_data[-1].replace("\r\n",'')
                x_train[k, i,0:41] = line_data[2:43]
                # motor actions start only after the sentence 
                
            # indicator of how many sequences we have gone through
            k = k+1 
        if k == numSeq:
            break
        
    dataFile.close()
        

    return x_train, y_train, numSeq, stepEachSeq, control_input


def plot(loss_list):
    #plt.subplot(2,3,1)
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)


def removeCases(y_train, control_input, numSeq, stepEachSeq, numControlNeurons):
    numSeqMod = numSeq - numSeq//10
    print(numSeqMod)
    y_mod = np.asarray(np.zeros((numSeqMod  , stepEachSeq)),dtype=np.int32)
    control_mod = np.asarray(np.zeros((numSeqMod, stepEachSeq, numControlNeurons)),dtype=np.float32)
    k = 0
    for i in range(numSeq):
        if i%10 != 0:
            y_mod[k,:] = y_train[i,:]
            control_mod[k,:,:] = control_input[i,:,:]
            k += 1
        else:
            print("removed sentence number ", i)

    return y_mod, control_mod, numSeqMod

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)

def filter_data_SOM(x_train, som, output_dim):
    new_x = np.asarray(np.zeros((len(x_train), len(x_train[0])  , output_dim)),dtype=np.float32)
    for i in range(len(x_train)):
        for t in range(len(x_train[0])):
            new_x[i, t, :] = np.reshape(som.map_vects(x_train[i]), [output_dim])
    return new_x

def shuffle_data(x_train, y_train, init_state, init_state_IO, control_mod):
    new_x = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    new_y = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2]))
    new_state = np.zeros((init_state.shape[0], init_state.shape[1]))
    new_state_IO = np.zeros((init_state_IO.shape[0], init_state_IO.shape[1]))
    new_control = np.zeros((control_mod.shape[0], control_mod.shape[1], control_mod.shape[2]))

    numSeq = x_train.shape[0]

    for i in range(numSeq):

        new_x[i,:,:] = x_train[i-1,:,:]

        new_y[i,:,:] = y_train[i-1,:,:]

        new_state[i,:] = init_state[i-1,:]

        new_state_IO[i,:] = init_state_IO[i-1,:]
    
        new_control[i, :, :] = control_mod[i-1,:,:]

    return new_x, new_y, new_state, new_state_IO, new_control

def create_batch(x_train, y_train, control_input, batch_size):
    x_out = np.zeros((batch_size, x_train.shape[1], x_train.shape[2]))
    y_out = np.zeros((batch_size, y_train.shape[1], y_train.shape[2]))
    control_out = np.zeros((batch_size, control_input.shape[1], control_input.shape[2]))
    for i in range(batch_size):
        seq_index = np.random.randint(0,y_train.shape[2]+1)
        print("sequence: ",seq_index)
        x_out[i, :, :] = x_train[seq_index, :, :]
        y_out[i, :, :] = y_train[seq_index, :, :]
        control_out[i, :, :] = control_input[seq_index, :, :]
    return x_out, y_out, control_out


x_train, y_train, numSeq, stepEachSeq, control_input = loadTrainingData()


print("data loaded")

#motor_input = 41 # I/O layer
lang_input = 100
output_neurons = 41
lang_dim1 = 160 # fast context
lang_dim2 = 45 # slow context 

old_x = x_train


#LEARNING_RATE = 5 * 1e-3
LEARNING_RATE = 0.005

NEPOCH = 5000 # number of times to train each sentence
#NEPOCH = 10

batch_size = 61
    
my_path= os.getcwd()
figure_path = os.path.join(my_path, "matrix/")


# DEFINE IF WE ARE TESTING MISSING SENTENCES HERE#
TEST_MISSING_SENTENCES = False
USING_BIG_BATCH = False
###################################################

MTRNN = CTRNNModel([lang_input, lang_dim1, lang_dim2], [2.0, 5.0, 60.0], stepEachSeq, lang_input, output_neurons, LEARNING_RATE)

y_train = np.zeros([numSeq, stepEachSeq, output_neurons], dtype=np.float32)
#y_train[:, :, :] = x_train[:, 1:-2, 0:output_neurons]
y_train[:,:,:] = np.roll(x_train, -1, axis=1)[:,:,0:output_neurons]
y_train[:,-1,:] = y_train[:,-2,:]

#x_train[:, 1:100, :] = x_train[:, 0:99, :] 

if TEST_MISSING_SENTENCES:
    y_mod, control_mod, numSeqmod =removeCases(y_train, control_input, numSeq, stepEachSeq, lang_dim2)
elif USING_BIG_BATCH:
    x_mod, y_mod, control_mod = create_batch(x_train, y_train, control_input, batch_size)
    numSeqmod = batch_size
else:
    y_mod = y_train
    x_mod = x_train
    control_mod = control_input
    numSeqmod = numSeq

#plt.ion()
#plt.figure()
#plt.show()
loss_list = []

threshold = 0.0005

MTRNN.sess.run(tf.global_variables_initializer())

standard_input = np.zeros([numSeqmod, stepEachSeq, lang_input], dtype = np.float32)

init_state_IO = np.zeros([numSeqmod, lang_input], dtype = np.float32)
init_state_fc = np.zeros([numSeqmod, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeqmod, lang_dim2], dtype = np.float32)
for i in range(numSeqmod):
    init_state_sc[i, :] = control_mod[i, 0, :] # store the initial state for each sequence != 0


best_loss = 10000000.0
epoch_idx = 0
print(np.shape(x_train))
raw_input()
while best_loss > threshold:
    print("Training epoch " + str(epoch_idx))
    x_mod, y_mod, init_state_sc, init_state_IO, control_mod = shuffle_data(x_mod, y_mod, init_state_sc, init_state_IO, control_mod)
    _total_loss, _train_op, _state_tuple, _logits = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple, MTRNN.logits], feed_dict={MTRNN.x:x_mod[0:numSeqmod], MTRNN.y:y_mod[0:numSeqmod], 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})
    total_loss = np.abs(_total_loss) / (numSeqmod*stepEachSeq) #if using MSE
    #total_loss = _total_loss # if using other means
    print("uncorrected loss: ", _total_loss)
    loss_list.append(total_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(total_loss))
    #plot(loss_list)
    if total_loss < best_loss: # only save when loss is lower
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(total_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss = total_loss
    epoch_idx += 1
    if epoch_idx > NEPOCH:
        break
    if np.isnan(_total_loss):
        print("nan found, stopping...")
        print(_logits)
        break

# TEST #
# TEST #

print("testing")
MTRNN.saver.restore(MTRNN.sess, save_path)


standard_input = np.zeros([1, stepEachSeq, lang_input], dtype = np.float32)

for t in range(numSeq):
    new_output = np.asarray(np.zeros((1, stepEachSeq, output_neurons)),dtype=np.float32)
    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_input)),dtype=np.float32)
    new_input[0, :, :] = x_train[t, :, :]
    new_output[0, :, :] = y_train[t, :, :]


    old_output = np.asarray(np.zeros((1, stepEachSeq, output_neurons)),dtype=np.float32)
    old_output[0, :, :] = y_train[t, :, 0:output_neurons]

    init_state_IO = np.zeros([1, lang_input], dtype = np.float32)
    init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
    init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)
    init_state_sc[0, :] = control_input[t, 0, :] # store the initial state for each sequence != 0

    _total_loss, _state_tuple, _logits = MTRNN.sess.run([MTRNN.total_loss, MTRNN.state_tuple, MTRNN.logits], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

    #print("control sequence: ", new_input[0, 0, :])

    output = _logits


    plt.ioff()
    plt.show()

    for i in range(lang_input):
        plt.plot(output[:, i], 'r')
        plt.plot(old_output[0, :, i], 'r-.')
        plt.show()


    #dataFile.close()

    total_error = 0.0
    for i in range(100):
        temp_error = 0.0
        for k in range(2):
            #print("data: ", data_softmax[i, k])
            #print("results: ", _softmax[i, k])
            temp_error += np.abs(old_output[0, i, k] - output[i, k])
        print("error: ", temp_error)
        total_error += temp_error

    print("total error: ", total_error)
    #sentence = ""
    #for i in range(30):
    #    for g in range(29):
    #        if _softmax[i,g] == max(_softmax[i]): #and max(_softmax[i]) > 0.6:
    #            if g <27:
    #                sentence += chr(96 + g)
    #            if g == 27:
    #                sentence += " "
    #            if g == 28:
    #                sentence += "."
    #print(sentence)

total_neurons = lang_input + lang_dim1 + lang_dim2 
print("this was done with ", NEPOCH, " epochs and a total of ", total_neurons, " neurons.")

#save weights#
MTRNN.saver.restore(MTRNN.sess, save_path)

U_input = np.zeros([lang_dim1 + lang_input, lang_input], dtype = np.float32)
U_fast = np.zeros([lang_dim1 + lang_input + lang_dim2, lang_dim1], dtype = np.float32)
U_slow = np.zeros([lang_dim1 + lang_dim2, lang_dim2], dtype = np.float32)


Weights = MTRNN.get_weights()
for v in Weights:
    temp_v = v.eval(MTRNN.sess)
    if (len(temp_v) == lang_dim1 + lang_input):
        U_input = temp_v
    if (len(temp_v) == lang_dim1 + lang_input + lang_dim2):
        U_fast = temp_v
    if (len(temp_v) == lang_dim1 + lang_dim2):
        U_slow = temp_v

totalinstances = len(U_input) + len(U_fast) + len(U_slow)
totalNeurons = lang_input + lang_dim1 + lang_dim2

baseline = 0
bigMatrix = np.zeros((totalinstances, totalNeurons))

bigMatrix[0:lang_input+lang_dim1, 0:lang_input] = U_input[:,:]
baseline += lang_input+lang_dim1
bigMatrix[baseline:baseline + lang_dim1 + lang_input + lang_dim2, lang_input:lang_input+lang_dim1] = U_fast[:,:]
baseline += lang_input+lang_dim1 + lang_dim2
bigMatrix[baseline:baseline + lang_dim1 + lang_dim2, lang_input+lang_dim1:lang_input + lang_dim1 + lang_dim2] = U_slow[:,:]

norm = np.max(abs(bigMatrix))

fig, ax = plt.subplots()
cax = ax.matshow(bigMatrix, cmap=plt.cm.seismic, vmin = -norm, vmax = norm)
cbar = fig.colorbar(cax, ticks = [-norm, 0, norm])
cbar.ax.set_yticklabels([str(-norm), '0', str(norm)])
plt.show()



plt.ioff()
plt.show()
MTRNN.sess.close()
#MTRNNTest.sess.close()

