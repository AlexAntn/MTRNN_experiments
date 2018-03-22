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

def get_sentence(verb, obj):
    verb = float(verb)
    obj = float(obj)
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

def get_combination(verb, obj, control_input):
    new_control = np.zeros((1, control_input.shape[1], control_input.shape[2]))
    verb = float(verb)
    obj = float(obj)
    if verb >= 0.0 and verb < 0.1:
        new_control[0, :, 0:4] = [0.0, 0.0, 0.0, 1.0]
    elif verb >= 0.1 and verb < 0.2:
        new_control[0, :, 0:4] = [0.0, 0.0, 1.0, 0.0]
    elif verb >= 0.2 and verb < 0.3:
        new_control[0, :, 0:4] = [0.0, 0.0, 1.0, 1.0]
    elif verb >= 0.3 and verb < 0.4:
        new_control[0, :, 0:4] = [0.0, 1.0, 0.0, 0.0]
    elif verb >= 0.4 and verb < 0.5:
        new_control[0, :, 0:4] = [0.0, 1.0, 0.0, 1.0]
    elif verb >= 0.5 and verb < 0.6:
        new_control[0, :, 0:4] = [0.0, 1.0, 1.0, 0.0]
    elif verb >= 0.6 and verb < 0.7:
        new_control[0, :, 0:4] = [0.0, 1.0, 1.0, 1.0]
    elif verb >= 0.7 and verb < 0.8:
        new_control[0, :, 0:4] = [1.0, 0.0, 0.0, 0.0]
    else:
        new_control[0, :, 0:4] = [1.0, 0.0, 0.0, 1.0]
    if obj >= 0.0 and obj < 0.1:
        new_control[0, :, 4:8] = [0.0, 0.0, 0.0, 1.0]
    elif obj >= 0.1 and obj < 0.2:
        new_control[0, :, 4:8] = [0.0, 0.0, 1.0, 0.0]
    elif obj >= 0.2 and obj < 0.3:
        new_control[0, :, 4:8] = [0.0, 0.0, 1.0, 1.0]
    elif obj >= 0.3 and obj < 0.4:
        new_control[0, :, 4:8] = [0.0, 1.0, 0.0, 0.0]
    elif obj >= 0.4 and obj < 0.5:
        new_control[0, :, 4:8] = [0.0, 1.0, 0.0, 1.0]
    elif obj >= 0.5 and obj < 0.6:
        new_control[0, :, 4:8] = [0.0, 1.0, 1.0, 0.0]
    elif obj >= 0.6 and obj < 0.7:
        new_control[0, :, 4:8] = [0.0, 1.0, 1.0, 1.0]
    elif obj >= 0.7 and obj < 0.8:
        new_control[0, :, 4:8] = [1.0, 0.0, 0.0, 0.0]
    else:
        new_control[0, :, 4:8] = [1.0, 0.0, 0.0, 1.0]

    return new_control


def loadTrainingData(input_neurons, output_neurons, numInputNeurons, numControlNeurons, stepEachSeq, numSeq):

    combinations = ["".join(seq) for seq in itertools.product("01", repeat = 2)]
    
    # sequence of vectors of encoders, to compare (x_train[0] corresponds to init_state)
    x_train = np.asarray(np.zeros((numSeq , stepEachSeq, input_neurons)),dtype=np.float32)

    # this is used for the softmax - will have to be changed?
    y_train = np.asarray(np.zeros((numSeq  , stepEachSeq)),dtype=np.int32)

    # control sequence working as input
    control_input = np.asarray(np.zeros((numSeq, stepEachSeq, numControlNeurons)),dtype=np.float32)
    

    print("steps: ", stepEachSeq)
    print("number of sequences: ", numSeq)
    #### Select true to pick random sequences to train (with repetitions!)####
    RANDOM_SEQUENCES = False

    dataFile = open("mtrnnTD.txt", 'r')

    totalSeq = 432
    sequences = []
    if RANDOM_SEQUENCES:
        for i in range(numSeq):
            sequences += [np.random.randint(0, totalSeq)]
            print(sequences[-1])
    else:
        sequences = np.arange(totalSeq)
    
    sentence_list = []
    verb_sequences = [["-1","-1"]]

    k = 0 #number of sequences
    t = -1 #number of saved sequences
    while True:
        line = dataFile.readline()
        if line == "":
            break
        if line.find("SEQUENCE") != -1:
            if k in sequences: # to select random sentences
                for i in range(4, stepEachSeq):
                    line = dataFile.readline()
                    line_data = line.split("\t")
                    line_data[-1] = line_data[-1].replace("\r\n",'')
                    if i==4:
                        # motor actions start only after the sentence 
                        t = t+1
                        x_train[t, 0,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, 0,output_neurons] = line_data[1]
                        x_train[t, 1,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, 1,output_neurons] = line_data[1]
                        x_train[t, 2,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, 2,output_neurons] = line_data[1]
                        x_train[t, 3,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, 3,output_neurons] = line_data[1]
                        x_train[t, 4,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, 4,output_neurons] = line_data[1]
                        control_input[t, :, :] = get_combination(line_data[0], line_data[1], control_input)
                        sentence = get_sentence(line_data[0], line_data[1])
                    else:
                        x_train[t, i,0:output_neurons] = line_data[2:output_neurons+2]
                        x_train[t, i,output_neurons] = line_data[1]
            # indicator of how many sequences we have gone through
            k = k+1 
        if k == totalSeq:
            break
        
    dataFile.close()
        

    return x_train, y_train, control_input, sentence_list


def plot(loss_list, fig, ax):
    ax.plot(loss_list, 'b')
    fig.canvas.flush_events()


def removeCases(y_train, x_train, control_input, numSeq, stepEachSeq, numControlNeurons, numCombinationsMiss):
    if numSeq%numCombinationsMiss == 0:
        numSeqMod = numSeq - numCombinationsMiss
        #print("number of sequences is multiple")
    else:
        numSeqMod = numSeq - numCombinationsMiss -1
        #print("number of sequences is NOT multiple")
    jumps = numSeq//numCombinationsMiss
    print("modified number of sequences: ", numSeqMod)
    y_mod = np.asarray(np.zeros((numSeqMod  , stepEachSeq, y_train.shape[2])),dtype=np.int32)
    x_mod = np.asarray(np.zeros((numSeqMod  , stepEachSeq, x_train.shape[2])),dtype=np.int32)
    control_mod = np.asarray(np.zeros((numSeqMod, stepEachSeq, numControlNeurons)),dtype=np.float32)
    k = 0
    for i in range(numSeq):
        if i%jumps != 0:
            y_mod[k,:, :] = y_train[i,:, :]
            x_mod[k, :, :] = x_train[i, :, :]
            control_mod[k,:,:] = control_input[i,:,:]
            k += 1
        else:
            print("removed sentence number ", i)
    return y_mod, x_mod, control_mod, numSeqMod, jumps

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
        seq_index = np.random.randint(0,y_train.shape[0])
        x_out[i, :, :] = x_train[seq_index, :, :]
        y_out[i, :, :] = y_train[seq_index, :, :]
        control_out[i, :, :] = control_input[seq_index, :, :]
    return x_out, y_out, control_out


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



####################################### control Variables #######################################
direction = True # generate sentence from CS
alternate = True
alpha = 0.5 
RUN_PCA = True
NEPOCH = 250000 # number of times to train each sequence

# DEFINE IF WE ARE TESTING MISSING SENTENCES HERE #
###################################################
TEST_MISSING_SENTENCES = False
numCombinationsMiss = 10

# DEFINE IF WE ARE USING BATCHED INPUTS HERE #
##############################################
USING_BIG_BATCH = True
batch_size = 32


threshold = 0.0005
best_loss_lang = 4.50
best_loss_cs = 4.0005
threshold_lang = 0.05
threshold_cs = 0.0001
lang_loss = 5
cs_loss = 5
lang_loss_list = [60.0]
cs_loss_list = [5.0]
loss_list = []

average_loss = 1000.0
best_loss = 1000.0

my_path= os.getcwd()

############# for testing - higher value, more sequences are jumped ##################
jumps = 1


###################################### Model Parameters #########################################
lang_input = 140 # size of IO layer
output_neurons = 41 # size of output/input sequence
input_neurons = 42 # size of input including objects
lang_dim1 = 160 # fast context
lang_dim2 = 45 # slow context 
control_dim = 8 # size of output/input control sequence

stepEachSeq = 104
numSeq = 432 #from dataset
LEARNING_RATE = 5 * 1e-4

MTRNN = CTRNNModel([lang_input, lang_dim1, lang_dim2], [2.0, 5.0, 60.0], stepEachSeq, lang_dim2, output_neurons, control_dim, input_neurons, LEARNING_RATE)


################################### acquire data ###############################################

x_train, y_train, control_input, sentence_list = loadTrainingData(input_neurons, output_neurons, lang_input, lang_dim2, stepEachSeq, numSeq)

old_x = x_train

########## Roll the outputs, so it tries predicting the future #############
y_train = np.zeros([numSeq, stepEachSeq, output_neurons], dtype=np.float32)
y_train[:,:,:] = np.roll(x_train, -1, axis=1)[:,:,0:output_neurons]
y_train[:,-1,:] = y_train[:,-2,:]

############These lines create an input only on step 0 ################################
#new_x_train = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
#new_x_train[:,:,0] = x_train[:,:,0]
#x_train[:,:,:] = new_x_train[:,:,:]
#######################################################################################


if TEST_MISSING_SENTENCES:
    y_mod, x_mod, control_mod, numSeqmod, jumps =removeCases(y_train, x_train, control_input, numSeq, stepEachSeq, lang_dim2, numCombinationsMiss)
elif USING_BIG_BATCH:
    x_mod, y_mod, control_mod = create_batch(x_train, y_train, control_input, batch_size)
    numSeqmod = batch_size
else:
    y_mod = y_train
    x_mod = x_train
    control_mod = control_input
    numSeqmod = numSeq

final_seq = np.zeros([numSeqmod, control_dim])
for i in range(numSeqmod):
    final_seq[i, :] = control_mod[i, 0, 0:control_dim]

init_state_IO = np.zeros([numSeqmod, lang_input], dtype = np.float32)
init_state_fc = np.zeros([numSeqmod, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([numSeqmod, lang_dim2], dtype = np.float32)

print("data loaded")

############################### training iterations ############################################
MTRNN.sess.run(tf.global_variables_initializer())

epoch_idx = 0
flag_saved = False
#complicated logic:
# 1) we train CS and motor(lang), or;
# 2) we train only motor(lang), or;
# 3) we train only CS.
while (alternate and (lang_loss_list[-1] > threshold_lang or cs_loss_list[-1] > threshold_cs)) or (not alternate and direction and lang_loss_list[-1] > threshold_lang) or (not alternate and not direction and cs_loss_list[-1] > threshold_cs):
    print("Training epoch " + str(epoch_idx))
    if USING_BIG_BATCH:
        x_mod, y_mod, control_mod = create_batch(x_train, y_train, control_input, batch_size)
        final_seq = np.zeros([numSeqmod, control_dim])
        for i in range(numSeqmod):
            final_seq[i, :] = control_mod[i, 0, 0:control_dim]
    if direction:
        cs_inputs = np.zeros([numSeqmod, stepEachSeq, lang_dim2], dtype = np.float32)
        cs_inputs = control_mod
    else:
        cs_inputs = np.zeros([numSeqmod, stepEachSeq, lang_dim2], dtype = np.float32)

    #################### Uncomment if you want to shuffle data for training #################
    #x_mod, y_mod, init_state_sc, init_state_IO, control_mod = shuffle_data(x_mod, y_mod, init_state_sc, init_state_IO, control_mod)
    #final_seq = np.zeros([numSeqmod, control_dim])
    #for i in range(numSeqmod):
    #    final_seq[i, :] = control_mod[i, 0, 0:control_dim]
    #########################################################################################


    t0 = datetime.datetime.now()
    _total_loss, _train_op = MTRNN.sess.run([MTRNN.total_loss, MTRNN.train_op], feed_dict={MTRNN.x:x_mod[0:numSeqmod], MTRNN.y:y_mod[0:numSeqmod], MTRNN.cs:cs_inputs, MTRNN.direction:direction, MTRNN.final_seq:final_seq, 'initU_0:0':init_state_IO, 'initC_0:0':init_state_IO, 'initU_1:0':init_state_fc, 'initC_1:0':init_state_fc, 'initU_2:0':init_state_sc, 'initC_2:0':init_state_sc})
    t1 = datetime.datetime.now()
    print("epoch time: ", (t1-t0).total_seconds())
    if direction:
        loss = _total_loss
        print("training sequences: ", loss)
        lang_loss = loss
        if loss > 150:
            lang_loss = 150
        lang_loss_list.append(loss)
    else:
        loss = _total_loss
        print("training CS: ", loss)
        cs_loss = loss
        if loss > 5:
            cs_loss = 5
        cs_loss_list.append(loss)
    if epoch_idx%2 == 0:
        average_loss = alpha*lang_loss + (1-alpha)*cs_loss
    loss_list.append(average_loss)
    print("Current best loss: ",best_loss)
    print("#################################")
    print("epoch "+str(epoch_idx)+", loss: "+str(loss))
    if (alternate and lang_loss_list[-1] < best_loss_lang and cs_loss_list[-1] < best_loss_cs) or (not alternate and direction and lang_loss_list[-1] < best_loss_lang) or (not alternate and not direction and cs_loss_list[-1] < best_loss_cs): 
        flag_saved = True
        best_loss = alpha*lang_loss_list[-1] + (1-alpha)*cs_loss_list[-1]
        model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(best_loss)
        save_path = MTRNN.saver.save(MTRNN.sess, model_path)
        best_loss_lang = lang_loss_list[-1]
        best_loss_cs = cs_loss_list[-1]
    epoch_idx += 1

    if alternate:
        if cs_loss_list[-1] < 1.5*lang_loss_list[-1] or cs_loss_list[-1] < threshold_cs:
            direction = True
            if epoch_idx%10 == 0:
                direction = not direction

        if lang_loss_list[-1] < 1.5*cs_loss_list[-1] or lang_loss_list[-1] < threshold_lang:
            direction = False
            if epoch_idx%10 == 0:
                direction = not direction

    t2 = datetime.datetime.now()
    print("saving time: ", (t2-t1).total_seconds())
    if epoch_idx > NEPOCH:
        break
    if np.isnan(_total_loss):
        print("nan found, stopping...")
        #print(_logits)
        break


##################################### Print error graph ####################################
plt.ion()
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.show()
plot(loss_list, fig, ax)
####################### FOR TEST PURPOSES ONLY ######################################
if not flag_saved:
    model_path = my_path + "/mtrnn_"+str(epoch_idx) + "_loss_" + str(average_loss)
    save_path = MTRNN.saver.save(MTRNN.sess, model_path)
#####################################################################################


########################################## TEST ############################################

print("testing - press enter to continue")
raw_input()
MTRNN.saver.restore(MTRNN.sess, save_path)

MTRNN.forward_step_test()

tf.get_default_graph().finalize()

init_state_IO = np.zeros([1, lang_input], dtype = np.float32)
init_state_fc = np.zeros([1, lang_dim1], dtype = np.float32)
init_state_sc = np.zeros([1, lang_dim2], dtype = np.float32)

verb = 0
plt.grid()
fullOutputList = []
fullErrorList = []
index_max = 0
prev_max = 0
for t in range(0,numSeq, jumps):
    new_output = np.asarray(np.zeros((1, stepEachSeq, output_neurons)),dtype=np.float32)
    new_cs = np.asarray(np.zeros((1, stepEachSeq, lang_dim2)),dtype=np.float32)
    new_input = np.asarray(np.zeros((1, stepEachSeq, input_neurons)),dtype=np.float32)
    new_final_seq = np.asarray(np.zeros((1, control_dim)),dtype=np.int32)
    new_cs[0, :, :] = control_input[t, :, :]
    new_input[0, :, :] = x_train[t, :, :]
    new_output[0, :, :] = y_train[t, :, :]

    new_final_seq[0, :] = control_input[t, 0, 0:control_dim]

    old_output = np.asarray(np.zeros((1, stepEachSeq, output_neurons)),dtype=np.float32)
    old_output[0, :, :] = y_train[t, :, 0:output_neurons]

########################################### True ##############################################
    direction=True

    States = np.zeros([stepEachSeq, 5, lang_dim1], dtype = np.float32) # 3 layers + Input + output
    output_list = []

    input_x = np.zeros([1, lang_dim2], dtype = np.float32)
    input_sentence = np.zeros([1, input_neurons], dtype = np.float32)
    State = ((init_state_IO, init_state_IO), (init_state_fc, init_state_fc), (init_state_sc, init_state_sc))
    ################################################
    
    for l in range(stepEachSeq):
        input_x[:,:] = new_cs[0,l,:]
        input_sentence[:,:] = new_input[0,l,:]
        init_state_00 = State[0][0]
        init_state_01 = State[0][1]
        init_state_10 = State[1][0]
        init_state_11 = State[1][1]
        init_state_20 = State[2][0]
        init_state_21 = State[2][1]
        outputs, new_state = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_x_t: input_x, MTRNN.Inputs_sentence_t: input_sentence,  'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20})
        output_list += [outputs]

        States[l, 0, 0:output_neurons] = States[l, 0, 0:output_neurons] + outputs[1][0][0:output_neurons]
        States[l, 1, 0:lang_input] = States[l, 1, 0:lang_input] + new_state[0][1]
        States[l, 2, 0:lang_dim1] = States[l, 2, 0:lang_dim1] + new_state[1][1]
        States[l, 3, 0:lang_dim2] = States[l, 3, 0:lang_dim2] + new_state[2][1]
        States[l, 4, 0:control_dim] = States[l, 4, 0:control_dim] + new_cs[0,l,0:control_dim]

    if RUN_PCA:
        execute_pca(sentence_list[t], t, States, output_neurons, lang_input, lang_dim1, lang_dim2, control_dim, direction)

    output_vec = np.zeros([stepEachSeq, output_neurons], dtype = np.float32)
    for i in range(len(output_list)):
        output_vec[i,:] = output_list[i][1][0][0:output_neurons]
    error_mat = np.abs(y_train[t, 4:104, :] - output_vec[4:104,:])
    average_error = np.zeros(100)
    average_error[:] = np.sum(error_mat, axis = 1)/output_neurons
    fullErrorList += [error_mat]
    fullOutputList += [output_vec]

    error = np.amax(error_mat)
    if error > prev_max:
        prev_max = error
        index_max = t

    for i in range(0, output_neurons, 1):
        color = i/output_neurons
        color_inv = 1 - color
        plt.figure(1)
        plt.semilogy(error_mat[:,i], color=(color_inv, color, 0.0))
    plt.semilogy(average_error, color=(0.0,0.0,1.0))
    verb += 1

########################################### False ###########################################
    direction=False

    old_output = np.asarray(np.zeros((1, stepEachSeq, output_neurons)),dtype=np.float32)
    old_output[0, :, :] = y_train[t, :, 0:output_neurons]

    States = np.zeros([stepEachSeq, 5, lang_dim1], dtype = np.float32) # 3 layers + Input + output
    state_list = []
    output_list = []

    input_x = np.zeros([1, lang_dim2], dtype = np.float32)
    input_sentence = np.zeros([1, input_neurons], dtype = np.float32)
    State = ((init_state_IO, init_state_IO), (init_state_fc, init_state_fc), (init_state_sc, init_state_sc))
    ################################################
    
    for l in range(stepEachSeq):
        input_sentence[:,:] = new_input[0,l,:]
        init_state_00 = State[0][0]
        init_state_01 = State[0][1]
        init_state_10 = State[1][0]
        init_state_11 = State[1][1]
        init_state_20 = State[2][0]
        init_state_21 = State[2][1]
        outputs, new_state = MTRNN.sess.run([MTRNN.outputs, MTRNN.new_state], feed_dict = {MTRNN.direction: direction, MTRNN.Inputs_x_t: input_x, MTRNN.Inputs_sentence_t: input_sentence,  'test/initU_0:0':init_state_01, 'test/initC_0:0':init_state_00, 'test/initU_1:0':init_state_11, 'test/initC_1:0':init_state_10, 'test/initU_2:0':init_state_21, 'test/initC_2:0':init_state_20})
        output_list += [outputs]

        States[l, 0, 0:output_neurons] = States[l, 0, 0:output_neurons] + new_input[0,l,0:output_neurons]
        States[l, 1, 0:lang_input] = States[l, 1, 0:lang_input] + new_state[0][1]
        States[l, 2, 0:lang_dim1] = States[l, 2, 0:lang_dim1] + new_state[1][1]
        States[l, 3, 0:lang_dim2] = States[l, 3, 0:lang_dim2] + new_state[2][1]
        States[l, 4, 0:control_dim] = States[l, 4, 0:control_dim] + outputs[0][0][0:control_dim]

    if RUN_PCA:
        execute_pca(sentence_list[t], t, States, output_neurons, lang_input, lang_dim1, lang_dim2, control_dim, direction)

    print("##############################################")
    print("Expected CS: ", new_final_seq[0, :])
    print("Output CS:", outputs[0][0][0:control_dim])
    print("##############################################")
    if t == 53 or t == 107 or t == 161 or t == 215 or t == 269 or t == 323 or t == 359 or t == 395 or t == 431:
        plt.figure(1)
        plt.axhline(y=0.1, ls='-', color='black', linewidth = 3.0)
        axes = plt.gca()
        axes.set_ylim([0.00001, 0.5])
        plt.savefig(my_path+'action' + str(t) + '_errorGraph.png', dpi=125)
        plt.clf()
        plt.grid()

        for i in range(0, output_neurons, 1):
            for g in range(0,stepEachSeq-4):
                if error_mat[g,i] == np.amax(error_mat):
                    print("error was on neuron " , i)
                    plt.figure(2)
                    plt.clf()
                    plt.plot(fullOutputList[index_max][4:104,i], 'r')
                    plt.plot(y_train[index_max, 4:104, i], 'b')
                    plt.savefig(my_path+'action' + str(t) + '_trajectory.png', dpi=125)
        index_max = 0
        prev_max = 0

total_neurons = lang_input + lang_dim1 + lang_dim2 
print("this was done with ", NEPOCH, " epochs and a total of ", total_neurons, " neurons.")

######################################## save weights #########################################
#MTRNN.saver.restore(MTRNN.sess, save_path)

#U_input = np.zeros([lang_dim1 + lang_input, lang_input], dtype = np.float32)
#U_fast = np.zeros([lang_dim1 + lang_input + lang_dim2, lang_dim1], dtype = np.float32)
#U_slow = np.zeros([lang_dim1 + lang_dim2, lang_dim2], dtype = np.float32)
##

#Weights = MTRNN.get_weights()
#for v in Weights:
#    print(v)
#    temp_v = v.eval(MTRNN.sess)
#    if (len(temp_v) == lang_dim1 + lang_input + lang_input):
#        U_input = temp_v
#    if (len(temp_v) == lang_dim1 + lang_input + lang_dim2):
#        U_fast = temp_v
#    if (len(temp_v) == lang_dim1 + lang_dim2):
#        U_slow = temp_v

#totalinstances = len(U_input) + len(U_fast) + len(U_slow)
#totalNeurons = lang_input + lang_dim1 + lang_dim2

#baseline = 0
#bigMatrix = np.zeros((totalinstances, totalNeurons))

#bigMatrix[0:lang_input+lang_dim1 + lang_input, 0:lang_input] = U_input[:,:]
#baseline += lang_input+lang_dim1 + lang_input
#bigMatrix[baseline:baseline + lang_dim1 + lang_input + lang_dim2, lang_input:lang_input+lang_dim1] = U_fast[:,:]
#baseline += lang_input+lang_dim1 + lang_dim2
#bigMatrix[baseline:baseline + lang_dim1 + lang_dim2, lang_input+lang_dim1:lang_input + lang_dim1 + lang_dim2] = U_slow[:,:]
#
#norm = np.max(abs(bigMatrix))
#
#fig, ax = plt.subplots()
#cax = ax.matshow(bigMatrix, cmap=plt.cm.seismic, vmin = -norm, vmax = norm)
#cbar = fig.colorbar(cax, ticks = [-norm, 0, norm])
#cbar.ax.set_yticklabels([str(-norm), '0', str(norm)])
#plt.show()

plt.ioff()
MTRNN.sess.close()

