#**********************
'''
Code name: X-BBox
Author: Anurag Kulshrestha
Purpose: Code for extracting training tiles and using U-Net and CNN-LSTM to learn
and classify sinkhole related fringe patterns in wrapped interferograms.
Author: Anurag Kulshrestha
Date created: 03-07-2022
Last modified: 02-02-2023

Info: 
1. The traning datasets are created using the XBBox method defined in function: make_training_tiles.
2. The training samples and labels are stored with file names beginnnig with 'trainX_' and 'train_Y' respectively.
3. The models are trained using TSx spotlight data, and tested on Sentinel-1 data.
4. For interferometric processing of TSx-spotlight data, please see TSx_spotlight.py
5. Functions for reading doris derived datasets are written in the doris_read_data.py file
6. The models are declared in models.py

'''
#**********************
import argparse, os, sys
import numpy as np

#***************
# self declared libraries
#***************
from doris_read_data import get_stack, get_dates
from models import 

#delcare plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
#delcare deep learning libraries
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, ConvLSTM2D, BatchNormalization, MaxPooling3D, Flatten, Conv3D, Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Conv3D, UpSampling3D, Reshape, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, Input, Model
from tensorflow.keras import layers
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

#Class for using functional programming to read and stack training tiles batchwise.
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_data_dir, list_IDs, dates, epochs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
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
        self.train_dir = train_data_dir
        self.dates = dates
        self.epochs = epochs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data using one of 1)UNet, 2)UNet_LSTM or 3) CNN_LSTM at a time
        #for UNet
        X, y = self.__data_generation_UNet(list_IDs_temp)
        #For CNN-LSTM
        #X, y = self.__data_generation_CNN_LSTM(list_IDs_temp, indexes)
        #For UNet-LSTM
        #X, y = self.__data_generation_UNet_LSTM(list_IDs_temp, indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    #data_generation function for UNet
    def __data_generation_UNet(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        win_size = self.dim[1]
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Read sample
            a_x = np.fromfile(self.train_dir + 'trainX' + ID, dtype = np.complex64)#np.load('data/' + ID + '.npy')
            a_x = a_x.reshape(a_x.size//win_size, win_size)[:win_size][...,np.newaxis]
            X[i,] = np.angle(a_x)

            # Read class
            a_y = np.fromfile(self.train_dir + 'trainY' + ID, dtype = np.float32)#self.labels[ID]
            y[i,] = a_y.reshape(a_y.size//win_size, win_size)[:win_size]

        #print(np.unique(y, return_counts=True))
        y=keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y
    
    
    #data_generation function for UNet-LSTM
    def __data_generation_UNet_LSTM(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, time_epochs, n_channels, *dim)
        win_size = self.dim[1]
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = np.float16)
        y = np.empty((self.batch_size), dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for j, date in enumerate(self.dates[:self.epochs]):
                # Read sample
                a_x = np.fromfile(self.train_dir + 'trainX' + ID[:-12] + date + '.raw', dtype = np.complex64)#np.load('data/' + ID + '.npy')
                a_x = a_x.reshape(a_x.size//win_size, win_size)[:win_size]
                X[i,...,j] = np.angle(a_x)
            
            # Read class
            a_y = np.fromfile(self.train_dir + 'trainY' + ID[:-12] + dates[self.epochs - 1]+ '.raw', dtype = np.float32)#self.labels[ID]
            y[i,] = a_y.reshape(a_y.size//win_size, win_size)[:win_size]
            #y[i] = self.labels[ID]

        #print(np.unique(y, return_counts=True))

        y=keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        X = X.transpose(0,3,1,2)[:,:,np.newaxis,...] #Transpose to reshape the data in the required input formatfor CNN-LSTM

        return X, y

    #data_generation function for CNN-LSTM
    def __data_generation_CNN_LSTM(self, list_IDs_temp, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, time_epochs ,n_channels, *dim)
        win_size = self.dim[1]
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = np.float16)
        #y = np.empty((self.batch_size, *self.dim), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        #print('self.epochs', self.dates[:self.epochs])
        # Generate data
        for i, (ID, index) in enumerate(zip(list_IDs_temp, indexes)):
            for j, date in enumerate(self.dates[:self.epochs]):
                # Read sample
                a_x = np.fromfile(self.train_dir + 'trainX' + ID[:-12] + date + '.raw', dtype = np.complex64)#np.load('data/' + ID + '.npy')
                a_x = a_x.reshape(a_x.size//win_size, win_size)[:win_size]#[:,np.newaxis,...]
                X[i,...,j] = np.angle(a_x)
                #print(ID)

            # Assign class
            y[i] = self.labels[index]


        y=keras.utils.to_categorical(y, num_classes=self.n_classes)

        #print(np.unique(y, return_counts=True))
        X = X.transpose(0,3,1,2)[:,:,np.newaxis,...]

        return X, y
    


# Function to implement XBBox methodology descriped in the paper
def make_training_tiles(doris_stack_dir, train_data_dir, train_data_ras_dir, dates, master_date):
    lines, pixels = get_slv_arr_shape(doris_stack_dir, dates[0])
    stride_l, stride_p = 500, 500
    #dates.remove('20160101')
    tile_size = 256
    
    #outer for both
    S1_outer_bbox = [2500, 2000, 9500, 6000] #[l0, p0, lN, pN]
    
    #for S1
    #S1_outer_bbox = [2700, 3500, 5600, 4900] #[l0, p0, lN, pN]
    #S1_inner_bbox = [3200, 3850, 5250, 4550]
    
    #for S2
    #S1_outer_bbox = [5500, 3500, 8600, 4900] #[l0, p0, lN, pN]
    S1_inner_bbox = [5900, 3900, 7400, 4500]
    
    #get lines and pixels
    inner_lines, inner_pixels = S1_inner_bbox[2] - S1_inner_bbox[0] +1, \
        S1_inner_bbox[3] - S1_inner_bbox[1] +1
    
    outer_lines, outer_pixels = S1_outer_bbox[2] - S1_outer_bbox[0] +1, \
        S1_outer_bbox[3] - S1_outer_bbox[1] +1
    
    midpoint = S1_inner_bbox[0] + inner_lines//2, S1_inner_bbox[1] + inner_pixels//2
    
    tile_radar_size_factors = np.arange(tile_size*1, max(outer_lines, outer_pixels), tile_size)
    #check which factors are closest in lines and pixels
    
    inner_min_fac_lp = np.argmin(np.absolute(tile_radar_size_factors - inner_lines)),\
        np.argmin(np.absolute(tile_radar_size_factors - inner_pixels))
    
    outer_min_fac_lp = np.argmin(np.absolute(tile_radar_size_factors - outer_lines)),\
        np.argmin(np.absolute(tile_radar_size_factors - outer_pixels))
    
    print(inner_min_fac_lp)
    print(outer_min_fac_lp)
    for line_fac in range(inner_min_fac_lp[0], outer_min_fac_lp[0]):
        for pix_fac in range(inner_min_fac_lp[1], outer_min_fac_lp[1]):
            lines = tile_radar_size_factors[line_fac]
            pixels = tile_radar_size_factors[pix_fac]
            
            l0_crop, lN_crop = midpoint[0] - lines//2, midpoint[0]+lines//2
            p0_crop, pN_crop = midpoint[1] - pixels//2, midpoint[1]+pixels//2
            
            #Translate l0_crop and then recalculate crop borders
            #Translate with stride
            l0_crop_tr = np.arange(l0_crop, S1_inner_bbox[0], stride_l)
            p0_crop_tr = np.arange(p0_crop, S1_inner_bbox[1], stride_p)
            #check for empty range  
            l0_crop_tr = np.array([l0_crop]) if len(l0_crop_tr)==0 else l0_crop_tr
            p0_crop_tr = np.array([p0_crop]) if len(p0_crop_tr)==0 else p0_crop_tr
            #recalculate end points
            lN_crop_tr = l0_crop_tr + lines
            pN_crop_tr = p0_crop_tr + pixels
            
            # loop for translation
            for l_0 in l0_crop_tr:
                for p_0 in p0_crop_tr:
                    l_N = l_0 + lines
                    p_N = p_0 + pixels
                    
                    #loop for mirroring
                    for mirror in ['O', 'X', 'Y','XY']:
                            #mirror='O'
                        for date in dates[2:]:# remove 1st date because it is the new master
                            #date = dates[2]
                            fold = os.path.join(doris_stack_dir, date)
                            #os.chdir(fold)
                            if mirror == 'O':
                                
                                mirror_cmd = ''
                            else:
                                mirror_cmd = '-m '+ mirror
                            
                            #Using the cpxfiddle class created by TUDelft to read, modify and store subsets of amplitude and InSAR phase images.
                            #train_x
                            os.system('cpxfiddle -w 7724 -e 0.3 -s 1.2 -q mixed -o sunraster -c jet -M {}/{} -f cr4 -l{} -p{} -L{} -P{} {} {}/cint_srp_rel1epoch.raw > {}/trainX_{}_{}_{}_{}_{}_{}.ras'.format(pix_fac+1, line_fac+1, l_0, p_0, l_N, p_N, mirror_cmd, fold, train_data_ras_dir, pix_fac+1, line_fac+1, l_0, p_0, mirror, date))
                            
                            os.system('cpxfiddle -w 7724 -q normal -o float -c jet -M {}/{} -f cr4 -l{} -p{} -L{} -P{} {} {}/cint_srp_rel1epoch.raw > {}/trainX_{}_{}_{}_{}_{}_{}.raw'.format(pix_fac+1, line_fac+1, l_0, p_0, l_N, p_N, mirror_cmd, fold, train_data_dir,pix_fac+1, line_fac+1, l_0, p_0, mirror, date))
                            
                            #train_y
                            os.system('cpxfiddle -w 7724 -q normal -o sunraster -c gray -M {}/{} -f i2 -l{} -p{} -L{} -P{} {} {}/label_i2.data > {}/trainY_{}_{}_{}_{}_{}_{}.ras'.format(pix_fac+1, line_fac+1, l_0, p_0, l_N, p_N, mirror_cmd, fold, train_data_ras_dir, pix_fac+1, line_fac+1,l_0, p_0, mirror, date))
                            
                            os.system('cpxfiddle -w 7724 -q normal -o float -M {}/{} -f i2 -l{} -p{} -L{} -P{} {} {}/label_i2.data > {}/trainY_{}_{}_{}_{}_{}_{}.raw'.format(pix_fac+1, line_fac+1, l_0, p_0, l_N, p_N, mirror_cmd, fold, train_data_dir,pix_fac+1, line_fac+1,l_0, p_0, mirror, date))
                    

#function to assign file indeices filenames
def get_list_ids(files):
    train_x_files = [i for i in files if i[5]=='X']
    list_IDs = [i[6:] for i in train_x_files]
    return list_IDs

#function to predict the sinkholes in S1 images
def test_on_S1(s1_tiles, model):
    print(s1_tiles.shape)
    pred = model.predict(s1_tiles)#(tiles_unet)
    print(pred.shape)
    return pred

#function to arrange cropped tiles back into an image
def recreate_tiles(rows ,cols , win_x, win_y, stride_x, stride_y, res_arr):
    tiles = []
    result = np.zeros((rows, cols))
    row_count=0
    for i in range(0, rows-win_y+1, stride_y):
        for j in range(0, cols-win_x+1, stride_x):
            result[i:i+win_y, j:j+win_x] = res_arr[row_count]
            #tiles.append(np.angle(tile))
            #plt.imshow(np.angle(tile[...,0]), cmap='jet')
            #plt.show()
            row_count+=1
    return result

#Function to carve Sentinel-1 tiles 
def carve_tiles(stack_arr, win_x, win_y, stride_x, stride_y):
    
    rows, cols, epochs = stack_arr.shape
    #tiles = np.empty((0, win_y, win_x, epochs))
    tiles = []
    row_count=0
    for i in range(0, rows-win_y+1, stride_y):
        for j in range(0, cols-win_x+1, stride_x):
            tile = stack_arr[i:i+win_y, j:j+win_x]
            tiles.append(np.angle(tile))
            #plt.imshow(np.angle(tile[...,0]), cmap='jet')
            #plt.show()
        row_count+=1
        print(i, i+win_y)
    return row_count,np.array(tiles)

#THIS IS THE MAIN FUNCITON
if __name__=='__main__':
    #Allocating GPU memory fraction
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .8
    session = InteractiveSession(config=config)
    
    #*********************************
    #*****   MAKE TRAINING DATA *************
    #*********************************
    doris_stack_dir = '/media/anurag/SSD_1/anurag/PhD_Project/Doris_Processing/Doris_Processing_35_wink_spotlight/stack'
    Train_Val_data_dir = '/home/anurag/Documents/PhDProject/Sentinel_Processing/paper_3/TSx_training_data'
    
    train_data_dir = Train_Val_data_dir+'/training_TSx_S2'
    train_data_ras_dir = Train_Val_data_dir+'/training_TSx_S2_ras'
    dates = sorted([i.split('/')[-1] for i,j,k in os.walk(doris_stack_dir) if len(i.split('/')[-1])==8])
    dates.remove('20160101') #the signals were not strong in this image
    dates = dates[2:] #the forst image was chosen as the new master
    master_date = '20151210'
    print('Dates:', dates)
    make_training_tiles(doris_stack_dir, train_data_dir, train_data_ras_dir, dates, master_date)
    
    #*********************************
    #*****   MAKE TEST DATA *************
    #*********************************
    doris_stack_dir_test_data = '/media/anurag/Seagate_badi_vaali/PhDProject/Doris_Processing/Doris_Processing_22_wink/new_datastack/stack'
    CROPPING = True
    CRP_LIST = [300,1430,12000,15000]
    tile_size = 256
    ifgs_arr = get_stack('ifgs', dates, doris_stack_dir_test_data, crop_switch=CROPPING, crop_list=CRP_LIST, sensor='s1', swath='1', burst='1')
    tiles_row_count, tiles = carve_tiles(ifgs_arr, tile_size, tile_size, tile_size, tile_size)
    tiles = tiles.transpose(0,3,1,2)[:,:,np.newaxis,:,:] #for shape consistent for CNN_LSTM
    
    np.save('tiles_gap_{}_epochs_12.npy'.format(str(tile_size)), tiles)
    
    #*********************************
    #*****   TRAINING  *************
    #*********************************
    
    #Declaring deep learning parameters
    tile_size = 256
    epochs = len(dates)
    #parameters for UNet
    params = {'dim': (tile_size,tile_size),
          'batch_size': 32,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
    
    #parameters for UNet-LSTM
    multi_temp_params = {'dim': (tile_size,tile_size),
          'batch_size': 32,
          'n_classes': 2,
          'n_channels': epochs,
          'shuffle': True}

    #****
    #Declaring file names for non-temporal modelling
    #****
    train_files = [k for i,j,k in os.walk(Train_Val_data_dir)][2]
    val_files = [k for i,j,k in os.walk(Train_Val_data_dir)][1]

    train_list_IDs = get_list_ids(train_files)
    val_list_IDs = get_list_ids(val_files)
    
    #****
    #Declaring filenames for temporal modelling. Uncomment for temporal modelling
    #****
    '''
    train_files_nosink = [k for i,j,k in os.walk(Train_Val_data_dir)][5] #this is for non-sinkhole examples for CNN-LSTM classification
    train_files = [k for k in train_files if k[-12:-4]==dates[0]] # this is done to extract the first time epoch of the stack
    val_files = [k for k in val_files if k[-12:-4]==dates[0]]

    train_list_IDs = get_list_ids(train_files)
    train_list_nosink_IDs = get_list_ids(train_files_nosink)
    val_list_IDs = get_list_ids(val_files)
    
    #Labels for CNN-LSTM modelling
    labels = np.append(np.zeros(len(train_files_nosink)), np.ones(len(train_files)))
    '''
    #****
    #Data generators
    #****
    #For U-Net
    training_generator = DataGenerator(Train_Val_data_dir+'/training_TSx_S2/', train_list_IDs, dates, epochs, 0, **params)
    validation_generator = DataGenerator(Train_Val_data_dir+'/training_TSx/', val_list_IDs, dates, epochs, 0, **params)
    
    #Please uncomment as per use
    #For UNet_LSTM
    #training_generator = DataGenerator(Train_Val_data_dir+'/training_TSx_S2/', train_list_IDs, dates, epochs, 0, **multi_temp_params)
    #validation_generator = DataGenerator(Train_Val_data_dir+'/training_TSx/', val_list_IDs, dates, epochs, 0, **multi_temp_params)
    
    #FOR CNN-LSTM
    #training_generator = DataGenerator(Train_Val_data_dir+'/cnn_lstm_train_data_2/', train_list_nosink_IDs + train_list_IDs, dates, epochs, labels, **multi_temp_params)
    

    
    #****
    #Model declaration
    #****
    #1. UNet model
    model = unet(pretrained_weights = None,input_size = (tile_size, tile_size , 1))
    #2. U-Net model with time epochs as features
    #model = unet(pretrained_weights = None,input_size = (tile_size, tile_size , epochs))
    
    #3. CNN-LSTM model
    '''
    model = define_model(epochs,
                 hidden_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 rows=tile_size, cols=tile_size,
                 #stateful=False,
                 num_classes=multi_temp_params['n_classes'],
                 bidirectional=False)
    '''
    
    #4. U-Net_LSTM model: This model did not converge, unfortunately
    '''
    model = define_model_upsample(epochs,
                 hidden_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 rows=tile_size, cols=tile_size,
                 #stateful=False,
                 num_classes=multi_temp_params['n_classes'],
                 bidirectional=False)
    '''
    
    model.fit(training_generator, epochs=10, validation_data=validation_generator)

    print(model.evaluate())
    
    model.save('UNet_TSx_train_S2')
    #model.save('CNN_LSTM_TSx_train_sink_nonsink_12epochs_bs_16')


    #*********************************
    #*****   PREDICTION  *************
    #*********************************
    
    #Test on S1
   
    s1_tiles = np.load('/home/kulshresthaa/PhDProject/paper_3/data/sentinel1_test_data/tiles_gap_20_epochs_12.npy')#.transpose(0,3,4,2,1)
    print(s1_tiles.shape)
    model = keras.models.load_model('CNN_LSTM_TSx_train_sink_nonsink_12epochs_bs_32')
    #model = keras.models.load_model('UNet_TSx_train_S2')
    pred = test_on_S1(s1_tiles, model)#, shp=1131, win_size=256)
    #patch_img = recreate_tiles(1131, 3001, 256,256,256,256, pred[...,0])
    tiles_row_count = (1131-256)//20+1
    patch_img_clstm = pred[:,1].reshape(tiles_row_count, pred.shape[0]//tiles_row_count)
    print(patch_img_clstm.shape)
    np.save('patch_img_CNN_LSTM_epochs_12.npy', patch_img_clstm)


    #*********************************
    #*****   Accuracy Assessment ******
    #*********************************
    
    #Define reference data
    ref_arr = np.zeros(*s1_tiles[:2])
    ref_arr[880:970, 1375:1500]  = 1  #bounding box covering the extermes of the sinkhole in Sentinel-1 crop
    
    #threshold probabiliy
    pred_arr = (patch_img_clstm>0.7).astype(int)
    
    #calculate F1 score
    print(precision_recall_fscore_support(ref_arr.flatten(), pred_arr.flaten(), average = 'weighted'))
