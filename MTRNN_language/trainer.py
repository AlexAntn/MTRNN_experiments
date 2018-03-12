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


import pandas as pd
from sklearn.decomposition import PCA

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

def loadTrainingData(numInputNeurons, numControlNeurons, stepEachSeq, numSeq):

    # sequence of letters
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, numInputNeurons)),dtype=np.float32)

    # control sequence
    y_train = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)

    control_input = np.asarray(np.zeros((numSeq, stepEachSeq, numControlNeurons)),dtype=np.float32)
    
    sentence_list = []

    lst = map(list, itertools.product([0, 1], repeat=4))
    k = 0
    for t in range(0,int(np.sqrt(numSeq)),1):
        for j in range(0,int(np.sqrt(numSeq)),1):
            sentence = get_sentence(lst[t], lst[j])
            sentence_list += [sentence]
            control_input[k, 0, 0:4] = lst[t]
            control_input[k, 0, 4:8] = lst[j]
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

    return x_train, y_train, control_input, sentence_list

def execute_pca(sentence, seq, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction):

    Mat_S0 = States[:,0, 0:lang_input]
    Mat_S1 = States[:,1, 0:input_layer]
    Mat_S2 = States[:,2, 0:lang_dim1]
    Mat_S3 = States[:,3, 0:lang_dim2]
    Mat_S4 = States[:,4, 0:control_dim]

#############################################

    inputdata = pd.DataFrame(data = Mat_S1)

    pca = PCA(n_components = 2)
    plotdata = pca.fit(Mat_S1).transform(Mat_S1)
    print("data explained by PCA for IO: ", pca.explained_variance_ratio_)

    for i in range(len(plotdata)):
        plt.scatter(plotdata[i,0], plotdata[i, 1], c=(0.0, 0.0, i/len(plotdata)), label = sentence, marker = 'o')
    plt.plot(plotdata[:,0], plotdata[:, 1], c='r', label = sentence)

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresIO/")

    plt.title("IO trajectory");
    plt.xlabel("PC1 :" + str(pca.explained_variance_ratio_[0]));
    plt.ylabel("PC2 :" + str(pca.explained_variance_ratio_[1]));
    plt.grid();
    if direction:
        plt.savefig(figure_path+sentence+'_IO_layer_' + str(seq) + '_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path+sentence+'_IO_layer_' + str(seq) + '_sentences_to_CS.png', dpi=125)
    plt.close()

#############################

    inputdata = pd.DataFrame(data = Mat_S2)

    pca = PCA(n_components = 2)
    plotdata = pca.fit(Mat_S2).transform(Mat_S2)
    print("data explained by PCA for FC: ", pca.explained_variance_ratio_)

    for i in range(len(plotdata)):
        plt.scatter(plotdata[i,0], plotdata[i, 1], c=(0.0, 0.0, i/len(plotdata)), label = sentence)
    plt.plot(plotdata[:,0], plotdata[:, 1], c='r', label = sentence)

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresFC/")

    plt.title("FC trajectory");
    plt.xlabel("PC1 :" + str(pca.explained_variance_ratio_[0]));
    plt.ylabel("PC2 :" + str(pca.explained_variance_ratio_[1]));
    plt.grid();
    if direction:
        plt.savefig(figure_path+sentence+'_FC_layer_'+ str(seq) + '_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path+sentence+'_FC_layer_'+ str(seq) + '_sentences_to_CS.png', dpi=125)
    plt.close()

#############################

    inputdata = pd.DataFrame(data = Mat_S3)

    pca = PCA(n_components = 2)
    plotdata = pca.fit(Mat_S3).transform(Mat_S3)
    print("data explained by PCA SC: ", pca.explained_variance_ratio_)

    for i in range(len(plotdata)):
        plt.scatter(plotdata[i,0], plotdata[i, 1], c=(0.0, 0.0, i/len(plotdata)), label = sentence)
    plt.plot(plotdata[:,0], plotdata[:, 1], c='r', label = sentence)

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresSC/")

    plt.title("SC trajectory");
    plt.xlabel("PC1 :" + str(pca.explained_variance_ratio_[0]));
    plt.ylabel("PC2 :" + str(pca.explained_variance_ratio_[1]));
    plt.grid();
    if direction:
        plt.savefig(figure_path+sentence+'_SC_layer_'+ str(seq) + '_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path+sentence+'_SC_layer_'+ str(seq) + '_sentences_to_CS.png', dpi=125)
    plt.close()

###########################################


def plot(loss_list, fig, ax):
    ax.plot(loss_list, 'b')
    fig.canvas.flush_events()



######################################### Control Variables ################################
direction = False
alpha = 0.5
RUN_PCA = True
NEPOCH = 80000 # number of times to train each sentence
threshold_lang = 0.015
threshold_cs = 0.0001
average_loss = 1000.0
best_loss = 5
best_loss_lang = 0.018
best_loss_cs = 0.0005

loss_list = []
lang_loss_list = [5.0]
cs_loss_list = [5.0]

my_path= os.getcwd()

########################################## Model parameters ################################
lang_input = 29 # size of output/input sentence
input_layer = 40 # IO layer
lang_dim1 = 160 # fast context
lang_dim2 = 45 # slow context (without control neurons)
control_dim = 8 # size of output/input control sequence

numSeq = 64
stepEachSeq = 30

LEARNING_RATE = 5 * 1e-3

MTRNN = CTRNNModel([input_layer, lang_dim1, lang_dim2], [2, 5, 60], stepEachSeq, lang_dim2, lang_input, control_dim, LEARNING_RATE)


#################################### acquire data ##########################################
x_train, y_train, control_input, sentence_list = loadTrainingData(lang_input, lang_dim2, stepEachSeq, numSeq)

final_seq = np.zeros([numSeq, control_dim])
for i in range(numSeq):
    final_seq[i,:] = control_input[i, 0, 0:8]

init_state_IO = np.zeros([numSeq, input_layer], dtype = np.float32)
init_state_fc = np.zeros([numSeq, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeq, lang_dim2], dtype = np.float32)

print("data loaded")

############################### training iterations #########################################

MTRNN.sess.run(tf.global_variables_initializer())

epoch_idx = 0
while lang_loss_list[-1] > threshold_lang and cs_loss_list[-1] > threshold_cs:
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
        print("training sentences: ", loss)
        new_loss = loss
        if loss > 5:
            new_loss = 5
        lang_loss_list.append(new_loss)
    else:
        loss = _total_loss
        print("training CS: ", loss)
        new_loss = loss
        if loss > 5:
            new_loss = 5
        cs_loss_list.append(new_loss)
    if epoch_idx%2 == 0:
        average_loss = alpha*lang_loss_list[-1] + (1-alpha)*cs_loss_list[-1]
    loss_list.append(average_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(loss))
    if lang_loss_list[-1] < best_loss_lang and cs_loss_list[-1] < best_loss_cs:
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss_lang = lang_loss_list[-1]
        best_loss_cs = cs_loss_list[-1]
        best_loss = alpha*lang_loss_list[-1] + (1-alpha)*cs_loss_list[-1]
    epoch_idx += 1

    if cs_loss_list[-1] < 2*lang_loss_list[-1] or cs_loss_list[-1] < threshold_cs:
        direction = True
        if epoch_idx%10 == 0:
            direction = not direction

    if lang_loss_list[-1] < 2*cs_loss_list[-1] or lang_loss_list[-1] < threshold_lang:
        direction = False
        if epoch_idx%10 == 0:
            direction = not direction

    t2 = datetime.datetime.now()
    print("saving time: ", (t2-t1).total_seconds())
    if epoch_idx > NEPOCH:
        break

##################################### Print error graph ####################################
plt.ion()
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.show()
plot(loss_list, fig, ax)
model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
save_path = MTRNN.saver.save(MTRNN.sess, model_path)

########################################## TEST ############################################

MTRNN.saver.restore(MTRNN.sess, save_path)
plt.ioff()
plt.show()
print("testing")

init_state_IO = np.zeros([1, input_layer], dtype = np.float32)
init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)

for i in range(0, numSeq, 1):
    new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    new_sentence = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
    new_final_seq = np.asarray(np.zeros((1, control_dim)), dtype=np.float32)
    new_input[0, :, :] = control_input[i, :, :]
    new_output[0, :] = y_train[i, :]
    
    new_final_seq[0,:] = control_input[i, 0, 0:8]


    direction = True

    if RUN_PCA:
        States = np.zeros([stepEachSeq, 5, lang_dim1], dtype = np.float32) # 3 layers + Input + output
        state_list = []
        output_list = []
        softmax_list = np.zeros([stepEachSeq, 29], dtype = np.float32)

        input_x = np.zeros([1, lang_dim2], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)
        State = MTRNN.zero_state_tuple(1)[1]
        with MTRNN.sess.as_default():
            for l in range(stepEachSeq):
                input_x[:,:] = new_input[0,l,:]
                input_sentence[:,:] = new_sentence[0,l,:]
                outputs, new_state, softmax = MTRNN.forward_step_test(input_x, input_sentence, State, direction)
                state_list += [new_state]
                output_list += [outputs]
                softmax_array = softmax.eval()
                softmax_list[l, :] = softmax_array
                State = new_state
                States[l, 0, 0:lang_input] = States[l, 0, 0:lang_input] + softmax_list[l,:]
                States[l, 1, 0:input_layer] = States[l, 1, 0:input_layer] + new_state[0][1].eval()
                States[l, 2, 0:lang_dim1] = States[l, 2, 0:lang_dim1] + new_state[1][1].eval()
                States[l, 3, 0:lang_dim2] = States[l, 3, 0:lang_dim2] + new_state[2][1].eval()
                States[l, 3, 0:control_dim] = States[l, 3, 0:control_dim] + new_input[0,l,0:8]
            execute_pca(sentence_list[i], i, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction)
            
        sentence = ""
        print("Sequence with new model:", new_input[:,0,0:8])
        for t in range(stepEachSeq):
            for g in range(29):
                if softmax_list[t,g] == max(softmax_list[t]): 
                    if g <27:
                        sentence += chr(96 + g)
                    if g == 27:
                        sentence += " "
                    if g == 28:
                        sentence += "."
        print(sentence)
        print("########################")
    else:
        _state_tuple, _softmax, _logits = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.softmax, MTRNN.logits], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

        sentence = ""
        print("Sequence with MTRNN:", new_input[:,0,0:8])
        for t in range(stepEachSeq):
            for g in range(29):
                if _softmax[t,g] == max(_softmax[t]): 
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

    if RUN_PCA:
        state_list = []
        output_list = []
        softmax_list = np.zeros([stepEachSeq, 29], dtype = np.float32)

        input_x = np.zeros([1, lang_dim2], dtype = np.float32)
        input_sentence = np.zeros([1, lang_input], dtype = np.float32)
        State = MTRNN.zero_state_tuple(1)[1]
        with MTRNN.sess.as_default():
            for l in range(stepEachSeq):
                input_x[:,:] = new_input[0,l,:]
                input_sentence[:,:] = new_sentence[0,l,:]
                #Inputs = [input_x, input_sentence]
                outputs, new_state, softmax = MTRNN.forward_step_test(input_x, input_sentence, State, direction)
                state_list += [new_state]
                output_list += [outputs]
                #print("size of new_state:", np.shape(new_state))
                softmax_array = softmax.eval()
                softmax_list[l, :] = softmax_array
                State = new_state
                output_array = outputs[0].eval()
                States[l, 0, 0:lang_input] = States[l, 0, 0:lang_input] + new_sentence[0,l,:]
                States[l, 1, 0:input_layer] = States[l, 1, 0:input_layer] + new_state[0][1].eval()
                States[l, 2, 0:lang_dim1] = States[l, 2, 0:lang_dim1] + new_state[1][1].eval()
                States[l, 3, 0:lang_dim2] = States[l, 3, 0:lang_dim2] + new_state[2][1].eval()
                States[l, 3, 0:control_dim] = States[l, 3, 0:control_dim] + output_array[0, 0:control_dim]
            execute_pca(sentence_list[i], i, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction)
        print("test: ", new_final_seq[0,:])
        print("output:", output_array[0, 0:8])
    else:
        _state_tuple, _logits_cs = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.logits_cs], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

        print("test: ", new_final_seq[0,:])
        print("output: ", _logits_cs[-1, 0:8])

MTRNN.sess.close()
#MTRNNTest.sess.close()

