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

#construction of control sequence (fixed combinations, 8 neurons, activation can be 0.0 or 1.0)
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
            sentence = get_sentence(lst[t+1], lst[j+1])
            sentence_list += [sentence]
            control_input[k, 0, 0:4] = lst[t+1]
            control_input[k, 0, 4:8] = lst[j+1]
            for f in range(0, stepEachSeq, 1):
                if f>=4 and f < len(sentence)+4:
                    if sentence[f-4] == ' ':
                        x_train[k, f,26] = 1
                        y_train[k, f] = 26
                    elif sentence[f-4] == '.':
                        x_train[k, f,27] = 1
                        y_train[k, f] = 27
                    else:
                        x_train[k, f, ord(sentence[f-4]) - 97] = 1
                        y_train[k, f] = ord(sentence[f-4]) - 97
                else:
                    y_train[k, f] = 26
            k = k+1 

    return x_train, y_train, control_input, sentence_list

def execute_pca(sentences, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, seqsTested, b):

    Mat_S0 = States[:, :,0, 0:lang_input]
    Mat_S1 = States[:, :,1, 0:input_layer]
    Mat_S2 = States[:, :,2, 0:lang_dim1]
    Mat_S3 = States[:, :,3, 0:lang_dim2]
    Mat_S4 = States[:, :,4, 0:control_dim]

#############################################

    component_1 = 0.0
    component_2 = 0.0
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S1[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S1[i]).transform(Mat_S1[i])
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], c=(color_inv, color, 0.0))
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresIO/")
    plt.title("IO trajectory - "+sentences[b])
    plt.xlabel("PC1 :" + str(component_1/i))
    plt.ylabel("PC2 :" + str(component_2/i))
    plt.grid()

    if direction:
        plt.savefig(figure_path + sentences[b] + '_IO_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path + sentences[b] + '_IO_layer_sentences_to_CS.png', dpi=125)
    plt.close()

#############################

    component_1 = 0.0
    component_2 = 0.0
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S2[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S2[i]).transform(Mat_S2[i])
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], c=(color_inv, color, 0.0))
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresFC/")
    plt.title("FC trajectory - "+sentences[b]);
    plt.xlabel("PC1 :" + str(component_1/i));
    plt.ylabel("PC2 :" + str(component_2/i));
    plt.grid();

    if direction:
        plt.savefig(figure_path+ sentences[b] + '_FC_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path+ sentences[b] + '_FC_layer_sentences_to_CS.png', dpi=125)
    plt.close()

#############################

    component_1 = 0.0
    component_2 = 0.0
    color = 0
    plt.subplot(111)
    plt.subplots_adjust(right = 0.75)
    for i in range(0,seqsTested,1):
        inputdata = pd.DataFrame(data = Mat_S3[i])

        color = i/seqsTested
        color_inv = 1 - color
        position = i/seqsTested
        pca = PCA(n_components = 2)
        plotdata = pca.fit(Mat_S3[i]).transform(Mat_S3[i])
        component_1 += pca.explained_variance_ratio_[0]
        component_2 += pca.explained_variance_ratio_[1]

        for t in range(len(plotdata)):
            plt.scatter(plotdata[t,0], plotdata[t, 1], c=(0.0, 0.0, t/len(plotdata)), marker = 'o')
        plt.plot(plotdata[:,0], plotdata[:, 1], color=(color_inv, color, 0.0))
        plt.text(1.05, position, sentences[i+b], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color=(color_inv, color, 0.0))

    my_path= os.path.dirname(__file__)
    figure_path = os.path.join(my_path, "figuresSC/")
    plt.title("SC trajectory - "+sentences[b]);
    plt.xlabel("PC1 :" + str(component_1/i));
    plt.ylabel("PC2 :" + str(component_2/i));
    plt.grid();

    if direction:
        plt.savefig(figure_path + sentences[b] + '_SC_layer_CS_to_sentences.png', dpi=125)
    else:
        plt.savefig(figure_path + sentences[b] + '_SC_layer_sentences_to_CS.png', dpi=125)
    plt.close()

###########################################


def plot(loss_list, fig, ax):
    ax.plot(loss_list, 'b')
    fig.canvas.flush_events()


######################################### Control Variables ################################
direction = True
alternate = True
alpha = 0.5
RUN_PCA = True
NEPOCH = 80000 # number of times to train each sentence
threshold_lang = 0.012
threshold_cs = 0.0001
average_loss = 1000.0
best_loss = 5
best_loss_lang = 0.015
best_loss_cs = 0.0005

loss_list = []
lang_loss_list = [5.0]
cs_loss_list = [5.0]

exclude_sentences = True # excludes one sentence from the list (currently, sentence #65)

my_path= os.getcwd()

jumps = 1

########################################## Model parameters ################################
lang_input = 28 # size of output/input sentence
input_layer = 40 # IO layer
lang_dim1 = 160 # fast context
lang_dim2 = 45 # slow context (without control neurons)
control_dim = 8 # size of output/input control sequence

numSeq = 81
stepEachSeq = 30

LEARNING_RATE = 5 * 1e-3

MTRNN = CTRNNModel([input_layer, lang_dim1, lang_dim2], [2, 5, 60], stepEachSeq, lang_dim2, lang_input, control_dim, LEARNING_RATE)


#################################### acquire data ##########################################
x_train, y_train, control_input, sentence_list = loadTrainingData(lang_input, lang_dim2, stepEachSeq, numSeq)

old_x = x_train
old_y = y_train
old_control = control_input
old_sentence = sentence_list
old_numSeq = numSeq

if exclude_sentences:
    numSeq = 80
    print(x_train.shape)
    new_x_train = np.zeros((80, x_train.shape[1], x_train.shape[2]))
    test_x = np.zeros((1, x_train.shape[1], x_train.shape[2]))
    new_x_train[:65] = x_train[:65]
    new_x_train[65:80] = x_train[66:81]
    test_x[0] = x_train[65]
    test_sentence = sentence_list[65]
    print(test_sentence)
    x_train = new_x_train
    new_y_train = np.zeros((80, y_train.shape[1]))
    test_y = np.zeros((1, y_train.shape[1]))
    new_y_train[:65] = y_train[:65]
    new_y_train[65:80] = y_train[66:81]
    test_y[0] = y_train[65]
    y_train = new_y_train
    new_control_input = np.zeros((80, control_input.shape[1], control_input.shape[2]))
    test_control = np.zeros((1, control_input.shape[1], control_input.shape[2]))
    new_control_input[:65] = control_input[:65]
    new_control_input[65:80] = control_input[66:81]
    test_control[0] = control_input[65]
    control_input = new_control_input

final_seq = np.zeros([numSeq, control_dim])
for i in range(numSeq):
    final_seq[i,:] = control_input[i, 0, 0:8]

init_state_IO = np.zeros([numSeq, input_layer], dtype = np.float32)
init_state_fc = np.zeros([numSeq, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeq, lang_dim2], dtype = np.float32)

print("data loaded")

############################### 
save_path = my_path + "/mtrnn_23054_loss_0.005006536259315908"
########################################## TEST ############################################

MTRNN.saver.restore(MTRNN.sess, save_path)
plt.ioff()
plt.show()
print("testing")

init_state_IO = np.zeros([1, input_layer], dtype = np.float32)
init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)

test_false = True
test_true = True

MTRNN.forward_step_test()

tf.get_default_graph().finalize()
States = np.zeros([10, stepEachSeq, 5, lang_dim1], dtype = np.float32) # 3 layers + Input + output

b=0
for i in range(0, 1, jumps):
    new_output = np.asarray(np.zeros((1, stepEachSeq)),dtype=np.int32)
    new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    new_sentence = np.asarray(np.zeros((1, stepEachSeq, lang_input)), dtype=np.float32)
    new_final_seq = np.asarray(np.zeros((1, control_dim)), dtype=np.float32)
    new_output[0, :] = test_y[0, :]
    
    new_final_seq[0,:] = test_control[0, 0, 0:8]

    print("sentence: ", sentence_list[65])


    if test_true:
        direction = True
        new_input[0, :, :] = test_control[0, :, :]
        if RUN_PCA:
            softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

            input_x = np.zeros([1, lang_dim2], dtype = np.float32)
            input_sentence = np.zeros([1, lang_input], dtype = np.float32)
            State = ((init_state_IO, init_state_IO), (init_state_fc, init_state_fc), (init_state_sc, init_state_sc))        
            ################################################
            
            for l in range(stepEachSeq):
                input_x[:,:] = new_input[0,l,:]
                input_sentence[:,:] = new_sentence[0,l,:]
                init_state_00 = State[0][0]
                init_state_01 = State[0][1]
                init_state_10 = State[1][0]
                init_state_11 = State[1][1]
                init_state_20 = State[2][0]
                init_state_21 = State[2][1]
                outputs, new_state, softmax = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state, MTRNN.softmax], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_x_t: input_x, MTRNN.Inputs_sentence_t: input_sentence,  'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20})

                softmax_list[l, :] = softmax
                State = new_state
                States[b, l, 0, 0:lang_input] = States[b, l, 0, 0:lang_input] + softmax_list[l,:]
                States[b, l, 1, 0:input_layer] = States[b, l, 1, 0:input_layer] + new_state[0][1]
                States[b, l, 2, 0:lang_dim1] = States[b, l, 2, 0:lang_dim1] + new_state[1][1]
                States[b, l, 3, 0:lang_dim2] = States[b, l, 3, 0:lang_dim2] + new_state[2][1]
                States[b, l, 4, 0:control_dim] = States[b, l, 4, 0:control_dim] + new_input[0,l,0:control_dim]

                
            sentence = ""
            print("Sequence with new model:", new_input[:,0,0:8])
            for t in range(stepEachSeq):
                for g in range(lang_input):
                    if softmax_list[t,g] == max(softmax_list[t]): 
                        if g <26:
                            sentence += chr(97 + g)
                        if g == 26:
                            sentence += " "
                        if g == 27:
                            sentence += "."
################################# Print table #####################################
            color = 0

            fig, ax = plt.subplots()
            Mat = np.transpose(softmax_list[:,0:lang_input])
            print(np.shape(Mat))
            cax = ax.matshow(Mat, cmap=plt.cm.binary, vmin = 0, vmax = 1)
            cbar = fig.colorbar(cax, ticks = [0, 1])
            cbar.ax.set_yticklabels(['0', '1'])
            for t in range(lang_input+1):
                ax.axhline(y=t+0.5, ls='-', color='black')
                if t < 26:
                    plt.text(-2,t+0.5,str(chr(97+t)))
                if t == 26:
                    plt.text(-2,t+0.5," ")
                if t == 27:
                    plt.text(-2,t+0.5,".")
            for t in range(stepEachSeq+1):
                ax.axvline(x=t+0.5, ls='-', color='black')
            plt.xlabel("timesteps");
            axes = plt.gca()
            axes.set_xlim([-0.5, 29.5])
            axes.set_ylim([27.5, -0.5])
            ax.set_yticklabels([])
            plt.show()
     
            print("####################################################")
            print("Output sentence: ",sentence)
        else:
            _state_tuple, _softmax, _logits = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.softmax, MTRNN.logits], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

            sentence = ""
            print("Sequence with MTRNN:", new_input[:,0,0:8])
            for t in range(stepEachSeq):
                for g in range(lang_input):
                    if _softmax[t,g] == max(_softmax[t]): 
                        if g <26:
                            sentence += chr(97 + g)
                        if g == 26:
                            sentence += " "
                        if g == 27:
                            sentence += "."
            print("####################################################")
            print("Output sentence: ",sentence)

    if test_false:
        direction = False
        new_input = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
        new_sentence[0, :, :] = test_x[0, :, :]

        if RUN_PCA:
            softmax_list = np.zeros([stepEachSeq, lang_input], dtype = np.float32)

            input_x = np.zeros([1, lang_dim2], dtype = np.float32)
            input_sentence = np.zeros([1, lang_input], dtype = np.float32)
            State = ((init_state_IO, init_state_IO), (init_state_fc, init_state_fc), (init_state_sc, init_state_sc))
            ################################################
            
            for l in range(stepEachSeq):
                input_x[:,:] = new_input[0,l,:]
                input_sentence[:,:] = new_sentence[0,l,:]
                init_state_00 = State[0][0]
                init_state_01 = State[0][1]
                init_state_10 = State[1][0]
                init_state_11 = State[1][1]
                init_state_20 = State[2][0]
                init_state_21 = State[2][1]
                outputs, new_state, softmax = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state, MTRNN.softmax], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_x_t: input_x, MTRNN.Inputs_sentence_t: input_sentence,  'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20})

                softmax_list[l, :] = softmax
                State = new_state
                States[b, l, 0, 0:lang_input] = States[b, l, 0, 0:lang_input] + new_sentence[0,l,:]
                States[b, l, 1, 0:input_layer] = States[b, l, 1, 0:input_layer] + new_state[0][1]
                States[b, l, 2, 0:lang_dim1] = States[b, l, 2, 0:lang_dim1] + new_state[1][1]
                States[b, l, 3, 0:lang_dim2] = States[b, l, 3, 0:lang_dim2] + new_state[2][1]
                States[b, l, 4, 0:control_dim] = States[b, l, 4, 0:control_dim] + outputs[0][0][0:control_dim]

            print("####################################################")
            print("Expected CS: ", test_control[0, 0, 0:8])
            print("Output CS:\n", outputs[0][0][0:control_dim])
            print("####################################################")
        else:
            _state_tuple, _logits_cs = MTRNN.sess.run([MTRNN.state_tuple, MTRNN.logits_cs], feed_dict={MTRNN.x:new_input, MTRNN.y:new_output, MTRNN.sentence:new_sentence, MTRNN.direction:direction, MTRNN.final_seq:new_final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})

            print("##############################################")
            print("expected CS: ", new_input[:,0,0:8])
            print("output CS: ", _logits_cs[-1, 0:control_dim])
            print("##############################################")
        print("\n")
        print("\n")
    b+= 1
    # for running PCA graphs
    if (i+1)%10 == 0 and i != 0:
        #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b, i-9)
        b = 0
        #execute_pca(old_sentence, States, lang_input, input_layer, lang_dim1, lang_dim2, control_dim, direction, b+1)


MTRNN.sess.close()

