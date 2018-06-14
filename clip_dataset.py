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
    def __init__(self, list_IDs, labels, batch_size=32, in_dim=(32,32,32), out_dim=(32,32,32), n_channels=1,
                 n_classes=10, every_nth_frame=1, shuffle=True, augment=False):
        'Initialization'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.every_nth_frame = every_nth_frame
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augment

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
            
    def resize_image(self, img):
        """Resize image to output shape in way that minimizes
        pixels lost due to differences in aspect ratios"""
        
        img_aspect_ratio = img.shape[1] / img.shape[0]
        out_aspect_ratio = self.out_dim[2] / self.out_dim[1]
        # scale to remove rotation artifacts
        scale = 1.06
        if img_aspect_ratio > out_aspect_ratio:
            # Image is wider than output, so scale to match height and crop excess width
            img = cv2.resize(img, (int(round(scale* self.out_dim[2] * img_aspect_ratio)), int(round(scale * self.out_dim[2]))))
        else:           
            # Image is taller than output, so scale to match width and crop excess heighth
            img = cv2.resize(img, (self.out_dim[1], int(round(self.out_dim[1] * img_aspect_ratio))))
        # Determine loactions to crop image
        col_start = int(round((img.shape[1] - self.out_dim[2]) / 2))
        col_end = col_start + self.out_dim[1]
        row_start = int(round((img.shape[0] - self.out_dim[1]) / 2))
        row_end = row_start + self.out_dim[2]
        # Crop image
        img = img[row_start:row_end, col_start:col_end, :]
        return img

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.out_dim, self.n_channels))
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
            if self.augment:
                flip_video = random() > 0.5
                # Translation
                rows, cols = self.in_dim
                shift_mag = 0.15
                shift_X = randint(-int(shift_mag * cols), int(shift_mag * cols))
                # shift_Y = randint(-int(shift_mag * rows), int(shift_mag * rows))
                shift_Y = 0
                
                # img = cv2.warpAffine(img,M,(cols,rows))
                # Rotation
                ang = uniform(-5, 5)
                
                # Random time-distance augmentation
                every_nth_frame = int(round(uniform(1, self.every_nth_frame)))

            else:
                flip_video = False
                rows, cols = self.in_dim
                shift_X = 0
                shift_Y = 0
                M_trans = np.float32([[1, 0, shift_X],[0, 1, shift_Y]])
                ang = 0
                M_rot = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), ang, 1)
                
                every_nth_frame = self.every_nth_frame
            
            # Set parameters on what frames to sample
            clip_length = 16
            vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))            
            
            if vid_length >= every_nth_frame * clip_length:
                
                # Read every nth frame
                start_frame = int(round(uniform(0, vid_length - (clip_length * every_nth_frame))))
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    if frames >= start_frame and frames < start_frame + clip_length * every_nth_frame:
                        if frames % every_nth_frame == 0:
                            if flip_video:
                                img = img[:, ::-1]
                            if self.augment:
                                # rows,cols,_ = img.shape
                                M_rot = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), ang, 1)
                                img = cv2.warpAffine(img, M_rot, (cols, rows))
                                M_trans = np.float32([[1, 0, shift_X],[0, 1, shift_Y]])
                                img = cv2.warpAffine(img, M_trans, (cols, rows))
                                # print(img.shape)
                            # col_start = int(round((self.in_dim[0] - self.out_dim[1]) / 2))
                            # col_end = col_start + self.out_dim[1]
                            # row_start = int(round((self.in_dim[1] - self.out_dim[2]) / 2))
                            # row_end = row_start + self.out_dim[2]
                            img = self.resize_image(img)
                            #vid.append(img[row_start:row_end, col_start:col_end, ::-1])
                            vid.append(img[:, :, ::-1])
                    elif frames >= start_frame + clip_length * every_nth_frame:
                        break
                    frames += 1
                vid = np.array(vid, dtype=np.float32)
                
            else:
                
                 # Read every frame
                start_frame = int(round(uniform(0, vid_length - (clip_length))))
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    if frames >= start_frame and frames < start_frame + clip_length:
                        if flip_video:
                            img = img[:, ::-1]
                        # col_start = int(round((self.in_dim[0] - self.out_dim[1]) / 2))
                        # col_end = col_start + self.out_dim[1]
                        # row_start = int(round((self.in_dim[1] - self.out_dim[2]) / 2))
                        # row_end = row_start + self.out_dim[2]
                        # print('{}:{}, {}:{}'.format(row_start, row_end, col_start, col_end))    
                        img = self.resize_image(img)
                        # vid.append(img[row_start:row_end, col_start:col_end, ::-1])
                        vid.append(img[:, :, ::-1])
                    frames += 1
                vid = np.array(vid, dtype=np.float32)

            # """
            # load mean and subtract
            mean_cube = np.load('models/train01_16_128_171_mean.npy')
            mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
            mean_cube = mean_cube[:, 8:120, 30:142, :]
            # print('mean_cube shape is: {}'.format(mean_cube.shape))
            # print('vid_slice shape is: {}'.format(len(vid_slice)))
            vid -= mean_cube[:len(vid), :, :, :] 
            # """             
                
            # Assigns image to random location in dataset
            #X[i,:, :, :] = im.astype('float32') / 255
            
            # Assigns video to random location in dataset
            try:
                X[i,:, :, :, :] = vid.astype('float32')
            except ValueError:
                print('vid_slice shape is: {}'.format(vid.shape))
                print('start_frame is: {}'.format(start_frame))
                print('frames is: {}'.format(frames))
                print('video is: \n{}'.format(ID))
            
            # Store label to random location in dataset
            y[i] = self.labels[ID]

        return (X, keras.utils.to_categorical(y, num_classes=self.n_classes))
    
    


