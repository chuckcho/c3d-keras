import numpy as np
import keras
import cv2
from cv2 import imread
import random
from random import random
from random import uniform
from random import randint

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Read image and switch from BGR to RGB
            # im = imread(ID)
            # im = im[:, :, ::-1]
            
            # Read video and subtract mean
            cap = cv2.VideoCapture(ID)
            vid = []
            frames = 0
            flip_video = random() > 0.5
            
            # Read all frames
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                frames += 1
                # vid.append(cv2.resize(im,None,dsize=[112, 200], interpolation = cv2.INTER_AREA))
                # Black out watermark
                # im[:100, 800:, :] = 0

                # im = cv2.resize(im,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

                # Data augmentation:
                # """
                # Horizontal flip
                if flip_video:
                    img = img[:, ::-1]
                # Translation
                # rows,cols,_ = im.shape
                # shift_mag = 0.1
                # shift_X = randint(-int(shift_mag * cols), int(shift_mag * cols))
                # shift_Y = randint(-int(shift_mag * rows), int(shift_mag * rows))
                # M = np.float32([[1, 0, shift_X],[0, 1, shift_Y]])
                # im = cv2.warpAffine(im,M,(cols,rows))
                # Rotation
                # ang = uniform(-10, 10)
                # M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), ang, 1)
                # im = cv2.warpAffine(im, M, (cols,rows))
                # """
                vid.append(img[:, :, ::-1])
            vid = np.array(vid, dtype=np.float32)


            clip_length = 16
            every_nth_frame = 8
            if frames > clip_length * every_nth_frame:
                # start_frame = int(frames / 2 - clip_length / 2) 
                start_frame = int(round(uniform(0, frames - (clip_length * every_nth_frame))))
                vid_slice = vid[start_frame:(start_frame + clip_length * every_nth_frame), :, :, :]
                vid_slice = vid_slice[::every_nth_frame, :, :, :]
            else:
                start_frame = int(round(uniform(0, frames - clip_length)))
                vid_slice = vid[start_frame:(start_frame + clip_length), :, :, :]


            # """
            # load mean and subtract
            mean_cube = np.load('models/train01_16_128_171_mean.npy')
            mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
            mean_cube = mean_cube[:, 8:120, 30:142, :]
            # print('mean_cube shape is: {}'.format(mean_cube.shape))
            # print('vid_slice shape is: {}'.format(len(vid_slice)))
            vid_slice -= mean_cube[:len(vid_slice), :, :, :] 
            # """
            
            

            
                
            # Assigns image to random location in dataset
            #X[i,:, :, :] = im.astype('float32') / 255
            
            # Assigns video to random location in dataset
            try:
                X[i,:, :, :, :] = vid_slice.astype('float32')
            except ValueError:
                print('vid_slice shape is: {}'.format(vid_slice.shape))
                print('start_frame is: {}'.format(start_frame))
                print('frames is: {}'.format(frames))
            
            # Store label to random location in dataset
            y[i] = self.labels[ID]

        return (X, keras.utils.to_categorical(y, num_classes=self.n_classes))
    
    


