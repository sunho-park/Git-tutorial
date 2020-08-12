def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    
    return x
    
def plain18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    
    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3))
    
    conv3_1 = conv2d_bn(conv2_3, 128, (3, 3), strides=2) # (28, 28, 128)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3))
    
    conv4_1 = conv2d_bn(conv3_2, 256, (3, 3), strides=2) # (14, 14, 256)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3))
    
    conv5_1 = conv2d_bn(conv4_2, 512, (3, 3), strides=2) # (7, 7, 512)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3))
    

    gap = GlobalAveragePooling2D()(conv5_2)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='Plain18')

    model.summary()    
    return model

def ResNet18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    
    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3), activation=None) # (56, 56, 64)
    
    shortcut_1 = Add()([conv2_3, conv2_1])
    shortcut_1 = Activation(activation='relu')(shortcut_1) # (56, 56, 64)

    
    conv3_1 = conv2d_bn(shortcut_1, 128, (3, 3), strides=2)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3)) # (28, 28, 128)
    
    shortcut_2 = conv2d_bn(shortcut_1, 128, (1, 1), strides=2, activation=None) # (56, 56, 64) -> (28, 28, 128)
    shortcut_2 = Add()([conv3_2, shortcut_2])
    shortcut_2 = Activation(activation='relu')(shortcut_2) # (28, 28, 128)

    
    conv4_1 = conv2d_bn(conv3_2, 256, (3, 3), strides=2)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3)) # (14, 14, 256)
    
    shortcut_3 = conv2d_bn(shortcut_2, 256, (1, 1), strides=2, activation=None) # (28, 28, 128) -> (14, 14, 256)
    shortcut_3 = Add()([conv4_2, shortcut_3])
    shortcut_3 = Activation(activation='relu')(shortcut_3) # (14, 14, 256)
    
    
    conv5_1 = conv2d_bn(conv4_2, 512, (3, 3), strides=2)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3)) # (7, 7, 512)
    
    shortcut_4 = conv2d_bn(shortcut_3, 512, (1, 1), strides=2, activation=None) # (14, 14, 256) -> (7, 7, 512)
    shortcut_4 = Add()([conv5_2, shortcut_4])
    shortcut_4 = Activation(activation='relu')(shortcut_4) # (7, 7, 512)
    

    gap = GlobalAveragePooling2D()(shortcut_4)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='ResNet18')
    
    model.summary()

    return model

    
if __name__ == '__main__':  
    main()
