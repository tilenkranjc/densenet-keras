
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def H_factory(x,nb_filters,growth_rate,bottleneck=False, weight_decay=1E-4):
    """ Tool to design H layers within each dense block

    x - input
    nb_filters - number of filters. Supplied from transitional layer if not compression is used.
    bottleneck - if True, another 1x1 conv layer will be added before 3x3 conv. If bottleneck is true, 
    growth rate has to be supplied as well.
    
    """
    
    x = BatchNormalization(axis=-1,
                            gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)
    
    x = Activation('relu')(x)

    # if bottleneck is True then add another 1x1 conv layer before 3x3 conv
    if bottleneck:
        x = Conv2D(filters=4*growth_rate,kernel_size=(1,1),
        kernel_initializer="he_uniform",
        padding="same", use_bias=False,
        kernel_regularizer=l2(weight_decay))(x)
    
    x = Conv2D(filters=nb_filters,kernel_size=(3,3),
        kernel_initializer="he_uniform",
        padding="same", use_bias=False,
        kernel_regularizer=l2(weight_decay))(x)
    
    return x
    


def dense_block(x,nb_layers,growth_rate,nb_filters,
    bottleneck=False, weight_decay=1E-4):

    """ Creates dense block by using H_factory.
    x - input (from previous layer)
    nb_layers - number of layers in dense block
    growth_rate - growth rate of network (see paper)
    nb_filters - number of initial filters
    bottleneck - add 1x1 layer before 3x3 layer.



    """
    
    # First create a list with only input feature maps. We'll add more in the loop below.
    list_of_features = [x]
    
    # Generate conv layers in dense block. After each layer add it's feature map to the list of feature
    # maps. This is then used as an input to the following layer.
    for i in range(nb_layers):
        cb = H_factory(x,growth_rate,growth_rate,bottleneck)
        list_of_features.append(cb)
        x = Concatenate(axis=-1)([x,cb])
        nb_filters += growth_rate
        
    # Return the final number of feature maps and the whole dense block
    return nb_filters,x


def transition_layer(x,nb_filters,compression_factor=1, weight_decay=1E-4):
    """ Transition layer, including batch norm, ReLU, 1x1 convolution and avg pool

    x - input (previous layer)
    nb_filters - number of filters to use. Take output from dense block and apply compression
    if needed
    """

    # use compression factor
    nb_filters = int(nb_filters//(1/compression_factor))

    x = BatchNormalization(axis=-1,
                            gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)
    
    x = Activation('relu')(x)

    x = Conv2D(filters=nb_filters,kernel_size=(1,1),
        kernel_initializer="he_uniform",
        padding="same", use_bias=False,
        kernel_regularizer=l2(weight_decay))(x)
    
    x = AveragePooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    return x

def DenseNet(img_dim,growth_rate,nb_classes,nb_filters,nb_layers,
    init_kernel_size=(3,3),
    bottleneck=False,
    compression_factor=1,
    weight_decay=1E-4):
    """ Function to generate the model of the DenseNet
    img_dim - dimensions of input images, tuple
    nb_classes - number of classes at the dense layer
    nb_filters - number of initial filters (usually 16)
    nb_layers - number of conv layers in each dense block, a list.

    """
    
    model_input = Input(shape=img_dim)
    
    x = Conv2D(filters=nb_filters,kernel_size=(3,3),
        kernel_initializer="he_uniform",
        padding="same", use_bias=False,
        name="initialConv2D",
        kernel_regularizer=l2(weight_decay))(model_input)
    # for ImageNet type of densenet we also need maxpooling
    #x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    # generate dense blocks
    for i,n in enumerate(nb_layers):
        nb_filters,x = dense_block(x,n,growth_rate,nb_filters,bottleneck)
        # last dense block doesn't have transition layer
        if i+1<len(nb_layers):
            x = transition_layer(x,nb_filters)
    
    # batch norm, ReLU after last dense block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # perform global average pool
    x = GlobalAveragePooling2D()(x)
    # FC layer with number of classes and softmax activation.
    x = Dense(nb_classes,activation='softmax')(x)
    
    # generate model with certain inputs and outputs
    model = Model(inputs=[model_input], outputs=[x], name="DenseNet")
    
    # return the model
    return model


def DenseNetIN(img_dim,growth_rate,nb_classes,nb_filters,nb_layers=[6,12,24,16],
    init_kernel_size=(7,7),
    bottleneck=True,
    compression_factor=0.5):
    """ Function to generate the model of the DenseNet
    img_dim - dimensions of input images, tuple
    nb_classes - number of classes at the dense layer
    nb_filters - number of initial filters (usually 16)
    nb_layers - number of conv layers in each dense block, a list.

    """
    
    model_input = Input(shape=img_dim)
    
    x = Conv2D(filters=nb_filters,kernel_size=init_kernel_size,padding="same")(model_input)
    # for ImageNet type of densenet we also need maxpooling
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    # generate dense blocks
    for i,n in enumerate(nb_layers):
        nb_filters,x = dense_block(x,n,growth_rate,nb_filters,bottleneck)
        # last dense block doesn't have transition layer
        if i+1<len(nb_layers):
            x = transition_layer(x,nb_filters)
    
    # batch norm, ReLU after last dense block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # perform global average pool
    x = GlobalAveragePooling2D()(x)
    # FC layer with number of classes and softmax activation.
    x = Dense(nb_classes,activation='softmax')(x)
    
    # generate model with certain inputs and outputs
    model = Model(inputs=[model_input], outputs=[x])
    
    # return the model
    return model