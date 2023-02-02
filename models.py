
# Function to define the U_Net model
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    print(model.summary())

    return model


# Function to define the CNN-LSTM model
def define_model(len_ts,
                 hidden_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 rows=256, cols=256,
                 num_classes=1,
                 bidirectional=False):
    
    inp = layers.Input(batch_shape= (batch_size, len_ts, nfeature, rows, cols),
                       name="input")  
    
    first_ConvLSTM = ConvLSTM2D(filters=16, kernel_size=(3, 3)
                       , data_format='channels_first'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True)(inp)
    
    first_BatchNormalization = BatchNormalization()(first_ConvLSTM)
    first_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(first_BatchNormalization)
    
    second_ConvLSTM = ConvLSTM2D(filters=32, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(first_Pooling)
    second_BatchNormalization = BatchNormalization()(second_ConvLSTM)
    second_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(second_BatchNormalization)
    
    
    
    third_ConvLSTM = ConvLSTM2D(filters=64, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(second_Pooling)
    
    third_BatchNormalization = BatchNormalization()(third_ConvLSTM)
    third_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(third_BatchNormalization)
    
    flatten = Flatten()(third_Pooling)
    
    dens = layers.Dense(num_classes, name="dense", activation='softmax')(flatten)
    model = models.Model(inputs=[inp],outputs=[dens])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary)
    return model

#Function to declare U-Net-LSTM model
def define_model_upsample(len_ts,
                 hidden_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 rows=256, cols=256,
                 num_classes=1,
                 bidirectional=False):
    
    inp = layers.Input(batch_shape= (batch_size, len_ts, nfeature, rows, cols),
                       name="input")  
    
    first_ConvLSTM = ConvLSTM2D(filters=16, kernel_size=(3, 3)
                       , data_format='channels_first'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True)(inp)
    
    first_BatchNormalization = BatchNormalization()(first_ConvLSTM)
    first_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(first_BatchNormalization)
    
    second_ConvLSTM = ConvLSTM2D(filters=32, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(first_Pooling)
    second_BatchNormalization = BatchNormalization()(second_ConvLSTM)
    second_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(second_BatchNormalization)
    
    third_ConvLSTM = ConvLSTM2D(filters=64, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(second_Pooling)
    
    third_BatchNormalization = BatchNormalization()(third_ConvLSTM)
    third_Pooling = MaxPooling3D(pool_size=(2, 4, 4), padding='same', data_format='channels_first')(third_BatchNormalization)
    
    
    
    up6 = ConvLSTM2D(filters=64, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(UpSampling3D(size = (2, 4, 4), data_format='channels_first')(third_Pooling))
    
    up7 = ConvLSTM2D(filters=32, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(UpSampling3D(size = (2, 4, 4), data_format='channels_first')(up6))
    
    up8 = ConvLSTM2D(filters=16, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(UpSampling3D(size = (2, 4, 4), data_format='channels_first')(up7))
    
    res = Reshape((16*len_ts, rows, cols))(up8)
    
    conv1 = Conv2D(2, 3, data_format='channels_first', activation = 'relu', padding = 'same')(res)
    conv10 = Conv2D(1, 1, data_format='channels_first', activation = 'sigmoid')(conv1)
    
    model = models.Model(inputs=[inp],outputs=[conv10])#[dens])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__=='__main__':
    pass
