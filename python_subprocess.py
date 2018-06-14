
# coding: utf-8

# In[1]:


# Standard Library Imports
import os
from os import listdir
from os.path import join
import sys
import random
from random import shuffle
from random import randint
import pickle

# Third-Party Imports
import keras
import keras.backend as K
from keras import optimizers
from keras import regularizers
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import ZeroPadding3D
from keras.layers.core import Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import cv2
import numpy as np

# Local Imports
import c3d_model
import clip_dataset
from clip_dataset import DataGenerator
import config_clips


# In[2]:


import tensorflow as tf
from keras import backend as k

i = int(sys.argv[1])

###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True

k.tensorflow_backend.set_session(tf.Session(config=config))


# In[3]:


# Use tf backend
dim_ordering = K.image_dim_ordering()
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering


# In[4]:



# In[5]:


def get_partitions(PATH):
    """
    Return dictionary that places each filename into
    a list with the parent dataset (train/valid/test) as the key
    """
    datasets = listdir(PATH)
    # print('datasets are: {}'.format(datasets))
    partitions = {d:[] for d in datasets}

    for d in datasets:
        classes = listdir(join(PATH, d))
        for c in classes:
            files = listdir(join(PATH, d, c))
            [partitions[d].append(join(PATH, d, c, f)) for f in files]
        # Randomize order
        shuffle(partitions[d])

    return partitions


# In[6]:


def get_best_model(model_dir, metric='acc'):
    """
    Return path to model weights with either lowest
    loss or highest accuracy
    """
    # Get all paths
    paths = listdir(model_dir)
    
    # Get only weight files
    weights = [p for p in paths if p[-5:] == '.hdf5']
    
    # Get only type of weights that were saved by desired metric
    weights = [w for w in weights if metric in w]

        
    vals = [float(w.rsplit('.hdf5', 1)[0].rsplit('-', 1)[-1]) for w in weights]
    if metric == 'acc':
        best_val = max(vals)
    else:
        best_val = min(vals)
        
    best_model = weights[vals.index(best_val)]
    return join(model_dir, best_model)


# In[7]:


def get_labels(PATH, classes_to_nums):
    """
    Return dictionary that places each filename into
    a list with the parent dataset as the key
    """
    datasets = listdir(PATH)
    print('datasets are: {}'.format(datasets))
    labels = {}

    for d in datasets:
        classes = listdir(join(PATH, d))
        for c in classes:
            files = listdir(join(PATH, d, c))
            num = classes_to_nums[c]
            temp = {join(PATH, d, c, f):num for f in files}
            labels = {**temp, **labels}

    return labels


# In[8]:


def load_model(dense_activation='relu'):
    pretrained_model_dir = './models'
    global backend

    print("[Info] Using backend={}".format(backend))

    model_weight_filename = join(pretrained_model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = join(pretrained_model_dir, 'sports1M_weights_tf.json')

    model = Sequential()
    input_shape=(16, 112, 112, 3) # l, h, w, c

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            border_mode='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten_1'))
    # FC layers group
    model.add(Dense(128, activation=dense_activation, name='fc6_lees'))
    model.add(Dropout(.5, name='dropout_1_lees'))
    model.add(Dense(128, activation=dense_activation, name='fc7_lees'))
    model.add(Dropout(.5, name='dropout_2_lees'))
    model.add(Dense(2, activation='softmax', name='fc8_lees'))

    print("[Info] Loading model weights...")    
    model.load_weights(model_weight_filename, by_name=True, skip_mismatch=True)
    print("[Info] Loading model weights -- DONE!")

    return model


# In[9]:


def run_and_save_inference_results(model, dataset_generator, path, trials=3):
    inference_results = []
    for i in range(trials):
        single_inference = model.evaluate_generator(generator=dataset_generator)
        inference_results.append(single_inference)
    pickle.dump(inference_results, open(path, "wb" ))
    return inference_results


# In[10]:


def run_verbose_inference(model, model_dir, training_generator, validation_generator, testing_generator):
    
    # Run inference on model as is, model with the best validation accuracy, and model with the best validation loss
    metric = 'final'
    run_and_save_inference_results(model, training_generator, join(model_dir, metric + '_training_results.pkl'), trials=1)
    run_and_save_inference_results(model, validation_generator, join(model_dir, metric + '_validation_results.pkl'), trials=5)
    run_and_save_inference_results(model, testing_generator, join(model_dir, metric + '_testing_results.pkl'), trials=5)

    metric = 'acc'
    best_model = get_best_model(model_dir, metric=metric)
    model.load_weights(best_model)
    run_and_save_inference_results(model, training_generator, join(model_dir, metric + '_training_results.pkl'), trials=1)
    run_and_save_inference_results(model, validation_generator, join(model_dir, metric + '_validation_results.pkl'), trials=5)
    run_and_save_inference_results(model, testing_generator, join(model_dir, metric + '_testing_results.pkl'), trials=5)

    metric = 'loss'
    best_model = get_best_model(model_dir, metric=metric)
    model.load_weights(best_model)
    run_and_save_inference_results(model, training_generator, join(model_dir, metric + '_training_results.pkl'), trials=1)
    run_and_save_inference_results(model, validation_generator, join(model_dir, metric + '_validation_results.pkl'), trials=5)
    run_and_save_inference_results(model, testing_generator, join(model_dir, metric + '_testing_results.pkl'), trials=5)

    
def get_callbacks(model_dir, model_iteration, patience=500):
    filepath_acc = join(
                    model_dir, 
                    "weights-acc-improvement-{epoch:03d}-{val_acc:.4f}.hdf5")
    checkpoint_acc = keras.callbacks.ModelCheckpoint(
                                                filepath_acc, 
                                                monitor='val_acc', 
                                                verbose=1, 
                                                save_best_only=True, 
                                                mode='max')
    filepath_loss = join(
                    model_dir, 
                    "weights-loss-improvement-{epoch:03d}-{val_loss:.4f}.hdf5")
    checkpoint_loss = keras.callbacks.ModelCheckpoint(
                                                filepath_loss, 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True, 
                                                mode='min')
    early_stopping = keras.callbacks.EarlyStopping(
                                                monitor='val_loss', 
                                                min_delta=0, 
                                                patience=patience, 
                                                verbose=0, 
                                                mode='auto')
    
    filepath_tb = join("..", "models", 'tensorboard_graphs', model_iteration)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=filepath_tb, histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint_acc, checkpoint_loss, early_stopping, tbCallBack]
    return callbacks_list


# ## Start of special over-the-weekend run

# In[11]:


PATH = config_clips.dataset_dir
classes_to_nums = config_clips.classes_to_nums
train_params = config_clips.train_params
valid_params = config_clips.valid_params
test_params = config_clips.test_params


# In[12]:
partition = get_partitions(PATH)
labels = get_labels(PATH, classes_to_nums)

# Generators
# Cycle through all possible datasets
all_possible_datasets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
val_folder = all_possible_datasets[i % 5]
test_folder = all_possible_datasets[(i + 1) % 5]
all_possible_datasets.pop(all_possible_datasets.index(val_folder))
all_possible_datasets.pop(all_possible_datasets.index(test_folder))

training_paths = []
for folder in all_possible_datasets:
    training_paths = training_paths + partition[folder]
    
training_generator = DataGenerator(training_paths, labels, **train_params)
validation_generator = DataGenerator(partition[val_folder], labels, **valid_params)
testing_generator = DataGenerator(partition[test_folder], labels, **test_params)


# In[13]:

print(i)
if i < 15:
    # loads unique names for specific training session
    model_iteration = 'model_c3d_0' + str(59 + i)
    model_dir = join("..", "models", model_iteration)
    model_name = join(model_dir, model_iteration + '.h5')
    history_name = join(model_dir, model_iteration + '_history.pkl')

    # makes new directory to place all saved files
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    # Callbacks
    callbacks_list = get_callbacks(model_dir=model_dir, model_iteration=model_iteration, patience=500)

    # picks an activation
    if i < 5:
        activation = 'relu'
    elif i < 10:
        activation = 'sigmoid'
    elif i < 15:
        activation = 'softmax'
    else:
        activation = 'relu'
    
    # loads a model
    model = load_model(dense_activation=activation)
    layers_to_train = ['fc6_lees', 'fc7_lees', 'fc8_lees']
    for layer in model.layers:
        if layer.name in layers_to_train:
            layer.trainable = True
            # print('{} IS trainable'.format(layer.name))
        else:
            layer.trainable = False
            # print('{} is NOT trainable'.format(layer.name))

    # compiles a model 
    adam = optimizers.adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # trains a model
    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=40,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=200,
                        initial_epoch=0,
                        workers=6)
    
    # Saves final model and training results
    # print('Saving model as {}'.format(model_name))
    model.save(model_name)
    with open(history_name, "wb" ) as f:
        pickle.dump(history.history, f)

    # Runs inference verbosely over datasets
    run_verbose_inference(model, model_dir, training_generator, validation_generator, testing_generator)


elif i < 25:
    # loads unique names for specific training session
    model_iteration = 'model_c3d_0' + str(59 + i)
    model_dir = join("..", "models", model_iteration)
    model_name = join(model_dir, model_iteration + '.h5')
    history_name1 = join(model_dir, model_iteration + '_history1.pkl')
    history_name2 = join(model_dir, model_iteration + '_history2.pkl')
    weights_name = join(model_dir, model_iteration + '.hdf5')

    # makes new directory to place all saved files
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Callbacks
    callbacks_list = get_callbacks(model_dir=model_dir, model_iteration=model_iteration, patience=500)
    
    
    # INTIAL training
    
    # loads a model
    model = load_model(dense_activation='softmax')
    layers_to_train = ['fc6_lees', 'fc7_lees', 'fc8_lees']
    for layer in model.layers:
        if layer.name in layers_to_train:
            layer.trainable = True
            # print('{} IS trainable'.format(layer.name))
        else:
            layer.trainable = False
            # print('{} is NOT trainable'.format(layer.name))

    # compiles a model 
    adam = optimizers.adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # trains a model
    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=40,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=150,
                        initial_epoch=0,
                        workers=6)
        
    # model.save(model_name)
    model.save_weights(weights_name)
    with open(history_name1, "wb" ) as f1:
        pickle.dump(history.history, f1)

    # RETRAINING
    # picks an activation
    if i < 20:
        activation = 'relu'
    elif i < 25:
        activation = 'sigmoid'
    else:
        activation = 'relu'
    
    # loads a model
    model = load_model(dense_activation=activation)
    layers_to_train = ['fc6_lees', 'fc7_lees', 'fc8_lees']
    for layer in model.layers:
        if layer.name in layers_to_train:
            layer.trainable = True
            # print('{} IS trainable'.format(layer.name))
        else:
            layer.trainable = False
            # print('{} is NOT trainable'.format(layer.name))

    # compiles a model 
    adam = optimizers.adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    model.load_weights(weights_name)
    # trains a model
    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=40,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=200,
                        initial_epoch=150,
                        workers=6)

    # Saves final model and training results
    # print('Saving model as {}'.format(model_name))
    model.save(model_name)
    with open(history_name2, "wb" ) as f2:
        pickle.dump(history.history, f2)

    # Runs inference verbosely over datasets
    run_verbose_inference(model, model_dir, training_generator, validation_generator, testing_generator)

else:

    # loads unique names for specific training session
    model_iteration = 'model_c3d_0' + str(26 + i)
    model_dir = join("..", "models", model_iteration)
    model_name = join(model_dir, model_iteration + '.h5')
    history_name1 = join(model_dir, model_iteration + '_history1.pkl')
    history_name2 = join(model_dir, model_iteration + '_history2.pkl')
    weights_name = join(model_dir, model_iteration + '.hdf5')

    # makes new directory to place all saved files
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Callbacks
    callbacks_list = get_callbacks(model_dir=model_dir, model_iteration=model_iteration, patience=500)
    
    
    # INTIAL training
    
    # loads a model
    model = load_model(dense_activation='softmax')
    layers_to_train = ['fc6_lees', 'fc7_lees', 'fc8_lees']
    for layer in model.layers:
        if layer.name in layers_to_train:
            layer.trainable = True
            # print('{} IS trainable'.format(layer.name))
        else:
            layer.trainable = False
            # print('{} is NOT trainable'.format(layer.name))

    # compiles a model 
    adam = optimizers.adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # trains a model
    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=40,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=20,
                        initial_epoch=0,
                        workers=6)
        
    # model.save(model_name)
    model.save_weights(weights_name)
    with open(history_name1, "wb" ) as f1:
        pickle.dump(history.history, f1)

    # RETRAINING
    # picks an activation
    if i < 30:
        activation = 'relu'
    else:
        activation = 'relu'
    
    # loads a model
    model = load_model(dense_activation=activation)
    layers_to_train = ['fc6_lees', 'fc7_lees', 'fc8_lees']
    for layer in model.layers:
        if layer.name in layers_to_train:
            layer.trainable = True
            # print('{} IS trainable'.format(layer.name))
        else:
            layer.trainable = False
            # print('{} is NOT trainable'.format(layer.name))

    # compiles a model 
    adam = optimizers.adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    model.load_weights(weights_name)
    # trains a model
    history = model.fit_generator(
                        generator=training_generator,
                        steps_per_epoch=20,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=400,
                        initial_epoch=20,
                        workers=6)

    # Saves final model and training results
    # print('Saving model as {}'.format(model_name))
    model.save(model_name)
    with open(history_name2, "wb" ) as f2:
        pickle.dump(history.history, f2)

    # Runs inference verbosely over datasets
    run_verbose_inference(model, model_dir, training_generator, validation_generator, testing_generator)

exit(0)
