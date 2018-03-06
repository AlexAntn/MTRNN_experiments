from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from CTRNN import CTRNNModel

import time
import operator
import io
import array
import datetime

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

#construction of control sequence (fixed combinations, 6 neurons, activation can be 0, 0.5 or 1.0)
def get_sentence(verb, obj):
    sentence = ""
    if verb == [0.0, 0.0, 0.0, 0.0]:
            verb_string = ""
    elif verb == [0.0, 0.0, 0.0, 1.0]:
            verb_string = "slide left"
    elif verb == [0.0, 0.0, 1.0, 0.0]:
            verb_string = "slide right"
    elif verb == [0.0, 0.0, 1.0, 1.0]:
            verb_string = "touch"
    elif verb == [0.0, 1.0, 0.0, 0.0]:
            verb_string = "reach"
    elif verb == [0.0, 1.0, 0.0, 1.0]:
            verb_string = "push"
    elif verb == [0.0, 1.0, 1.0, 0.0]:
            verb_string = "pull"
    elif verb == [0.0, 1.0, 1.0, 1.0]:
            verb_string = "point at"
    elif verb == [1.0, 0.0, 0.0, 0.0]:
            verb_string = "grasp"
    elif verb == [1.0, 0.0, 0.0, 1.0]:
            verb_string = "lift"
    if obj == [0.0, 0.0, 0.0, 0.0]:
            obj_string = ""
    elif obj == [0.0, 0.0, 0.0, 1.0]:
            obj_string = "tractor"
    elif obj == [0.0, 0.0, 1.0, 0.0]:
            obj_string = "hammer"
    elif obj == [0.0, 0.0, 1.0, 1.0]:
            obj_string = "ball"
    elif obj == [0.0, 1.0, 0.0, 0.0]:
            obj_string = "bus"
    elif obj == [0.0, 1.0, 0.0, 1.0]:
            obj_string = "modi"
    elif obj == [0.0, 1.0, 1.0, 0.0]:
            obj_string = "car"
    elif obj == [0.0, 1.0, 1.0, 1.0]:
            obj_string = "cup"
    elif obj == [1.0, 0.0, 0.0, 0.0]:
            obj_string = "cubes"
    elif obj == [1.0, 0.0, 0.0, 1.0]:
            obj_string = "spiky"
    if obj_string != "" and verb_string != "":
        sentence = verb_string + " the " + obj_string + "."
    else:
        sentence = verb_string + obj_string +"."
    return sentence

######################################################################################
# This function loads data from a file, to train the network
# inputs are sequential (and always same order). 
def loadTrainingData():


    numInputNeurons = 29 # language neurons, 26 for letters plus space and stop plus "nothing"
    numControlNeurons = 45 
    stepEachSeq = 30


    #sentences = ["bafffffffff", "kceeeeeeeee", "vvvvvrrrrr", "ghllllll", "uuuutttttt", "push the ball"]
    numSeq = 100
    
    # sequence of letters
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, numInputNeurons)),dtype=np.float32)

    # control sequence
    y_train = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)

    control_input = np.asarray(np.zeros((numSeq, stepEachSeq, numControlNeurons)),dtype=np.float32)
    
    lst = map(list, itertools.product([0, 1], repeat=4))
    k = 0
    for t in range(0,10,1):
        for j in range(0,10,1):
            sentence = get_sentence(lst[t], lst[j])
            #print(sentence)
            control_input[k, 0, 0:4] = lst[t]
            control_input[k, 0, 4:8] = lst[j]
            #print("control:", control_input[k, 0, 0:8]) 
            #raw_input()
            for f in range(0, stepEachSeq, 1):
                if f>=4 and f < len(sentence)+4:
                    if sentence[f-4] == ' ':
                        x_train[k, f,26] = 1
                        y_train[k, f] = 27
                    elif sentence[f-4] == '.':
                        x_train[k, f,27] = 1
                        y_train[k, f] = 28
                    else:
                        x_train[k, f, ord(sentence[f-4]) - 97] = 1
                        y_train[k, f] = ord(sentence[f-4]) - 96
                else:
                    y_train[k, f] = 27
            k = k+1 



    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)
    
    #control_input[0, 0, :] = [0.0, 0.0, 0.0]
 #   control_input[0, 0, 0:3] = [0.0, 0.0, 1.0]
 #   control_input[1, 0, 0:3] = [0.0, 1.0, 0.0]
#    control_input[2, 0, 0:3] = [0.0, 1.0, 1.0]
#    control_input[3, 0, 0:3] = [1.0, 0.0, 0.0]
#    control_input[4, 0, 0:3] = [1.0, 0.0, 1.0]
#    control_input[5, 0, 0:3] = [1.0, 1.0, 0.0]

    #k = 0 #number of sequences
    #for sentence in sentences:
    #    
    #    for f in range(0, stepEachSeq, 1):
    #        if f>=4 and f < len(sentence)+4:
     #           if sentence[f-4] == ' ':
    #                x_train[k, f,26] = 1
    #                y_train[k, f] = 27
    #            elif sentence[f-4] == '.':
    #                x_train[k, f,27] = 1
    #                y_train[k, f] = 28
    #            else:
    #                x_train[k, f, ord(sentence[f-4]) - 97] = 1
    #                y_train[k, f] = ord(sentence[f-4]) - 96
    #        else:
    #            y_train[k, f] = 27
    #    k = k+1 

    return x_train, y_train, numSeq, stepEachSeq, control_input


def plot(loss_list, fig, ax):
    #fig.cla()
    ax.plot(loss_list, 'b')

    fig.canvas.flush_events()
    #plt.draw()
    #plt.pause(0.0001)


x_train, y_train, numSeq, stepEachSeq, control_input = loadTrainingData()


print("data loaded")

lang_input = 29 # I/O layer
input_layer = 40
lang_dim1 = 160 # fast context
lang_dim2 = 45 # slow context (without control neurons)
control_dim = 8 # control neurons

LEARNING_RATE = 5 * 1e-3

NEPOCH = 80000 # number of times to train each sentence
    
my_path= os.getcwd()
figure_path = os.path.join(my_path, "matrix/")


MTRNN = CTRNNModel([input_layer, lang_dim1, lang_dim2], [2, 5, 60], stepEachSeq, lang_dim2, lang_input, control_dim, LEARNING_RATE)


plt.ion()
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.show()

loss_list = []
lang_loss_list = [5.0]
cs_loss_list = [5.0]

threshold = 0.0005

MTRNN.sess.run(tf.global_variables_initializer())

print("control sequences:", control_input[:,0,0:8])
raw_input()

final_seq = np.zeros([numSeq, control_dim])
for i in range(numSeq):
    final_seq[i,:] = control_input[i, 0, 0:8]


init_state_IO = np.zeros([numSeq, input_layer], dtype = np.float32)
init_state_fc = np.zeros([numSeq, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeq, lang_dim2], dtype = np.float32)
#init_state_cn = np.zeros([numSeq, control_dim], dtype = np.float32)

######################################### Control Variables ################################
direction = True
test = False
alpha = 1

average_loss = 1000.0

best_loss = 0.07
epoch_idx = 0
while best_loss > threshold:
    print("Training epoch " + str(epoch_idx))
    if direction:
        inputs = np.zeros([numSeq, stepEachSeq, lang_input], dtype = np.float32)
        cs_inputs = control_input
    else:
        inputs = x_train
        cs_inputs = np.zeros([numSeq, stepEachSeq, lang_dim2], dtype = np.float32)

    t0 = datetime.datetime.now()
    _total_loss, _train_op, _state_tuple = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op, MTRNN.state_tuple], feed_dict={MTRNN.x:cs_inputs, MTRNN.y:y_train, MTRNN.sentence:inputs, MTRNN.direction:direction, MTRNN.final_seq:final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})
    t1 = datetime.datetime.now()
    print("epoch time: ", (t1-t0).total_seconds())
    if direction:
        loss = _total_loss
        new_loss = loss
        if loss > 5:
            new_loss = 5
        lang_loss_list.append(new_loss)
    else:
        loss = _total_loss
        new_loss = loss
        if loss > 5:
            new_loss = 5
        cs_loss_list.append(new_loss)
    if epoch_idx%2 == 0:
        average_loss = alpha*lang_loss_list[-1] + (1-alpha)*cs_loss_list[-1]
    loss_list.append(_total_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(loss))
    #plot(loss_list, fig, ax)
    if average_loss < best_loss and lang_loss_list[-1] < 0.010: # only save when loss is lower
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss = average_loss
    epoch_idx += 1

    if cs_loss_list[-1] < 2*lang_loss_list[-1]:
        direction = True
        if epoch_idx%10 == 0:
            direction = not direction

    if lang_loss_list[-1] < 2*cs_loss_list[-1] or lang_loss_list[-1] < 0.010:
        direction = False
        if epoch_idx%10 == 0:
            direction = not direction

    t2 = datetime.datetime.now()
    print("saving time: ", (t2-t1).total_seconds())
    if epoch_idx > NEPOCH:
        break
plot(loss_list, fig, ax)

# TEST #

MTRNN.saver.restore(MTRNN.sess, save_path)
plt.ioff()
plt.show()
print("testing")
test=True


init_state_IO = np.zeros([1, input_layer], dtype = np.float32)
init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)
#init_state_cn = np.zeros([1, control_dim], dtype = np.float32)

for i in range(0, numSeq, 1):
    new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    new_sentence = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
    new_final_seq = np.asarray(np.zeros((1, control_dim)), dtype=np.float32)
    new_input[0, :, :] = control_input[i, :, :]
    new_output[0, :] = y_train[i, :]
    #print(new_output)
    #print(new_input)
    
    new_final_seq[0,:] = control_input[i, 0, 0:8]

    #if i == 0:
    #    new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    #    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    #    new_input[0, :, 0:8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    #    #new_output[0, :] = y_train[i, :]

    direction = True

    _state_tuple, _softmax, _logits = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.softmax, MTRNN.logits], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

    sentence = ""
    print("Sequence:", new_input[:,0,0:8])
    for i in range(stepEachSeq):
        for g in range(29):
            if _softmax[i,g] == max(_softmax[i]): 
                if g <27:
                    sentence += chr(96 + g)
                if g == 27:
                    sentence += " "
                if g == 28:
                    sentence += "."
    print(sentence)
    print("########################")


    direction = False
    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    new_sentence[0, :, :] = x_train[i, :, :]

    _state_tuple, _logits_cs = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.logits_cs], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

    print("output: ", _logits_cs[-1, 0:8])
    print("test: ", new_final_seq[0,:])

MTRNN.sess.close()
#MTRNNTest.sess.close()

