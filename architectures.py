from densenet import DenseNet, DenseNetIN

""" Architectures of DenseNet as described in the paper:
https://arxiv.org/pdf/1608.06993.pdf

Name legend:
DenseNet40k12 - 40 layers (12,12,12), k=12
DenseNet100k12 - 100 layers (32,32,32), k=12
DenseNet100k24 - 100 layers (32,32,32), k=24
DenseNetBC100k12 - 100 layers (32,32,32), k=12, bottleneck True and compression 0.5
DenseNetBC250k24 - 250 layers (82,82,82), k=12, bottleneck True and compression 0.5
DenseNetBC190k40 - 190 layers (62,62,62), k=12, bottleneck True and compression 0.5
DenseNetIN121 - 121 layers (6,12,24,16), k=32, bottleneck True, compression 0.5, additional max pool
DenseNetIN169 - 169 layers (6,12,32,32), k=32, bottleneck True, compression 0.5, additional max pool
DenseNetIN201 - 201 layers (6,12,48,32), k=32, bottleneck True, compression 0.5, additional max pool
DenseNetIN264 - 264 layers (6,12,64,48), k=32, bottleneck True, compression 0.5, additional max pool

Each network has default set the defining parameters. You only need to define img_dim and nb_classes.

"""

def DenseNet40k12(img_dim,  nb_classes, growth_rate=12, nb_filters=16, 
    nb_layers=[12,12,12],
    init_kernel_size=(3,3),
    bottleneck=False,
    compression_factor=1,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNet100k12(img_dim, nb_classes, growth_rate=12, nb_filters=16, 
    nb_layers=[32,32,32],
    init_kernel_size=(3,3),
    bottleneck=False,
    compression_factor=1,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNet100k24(img_dim, nb_classes, growth_rate=24, nb_filters=16, 
    nb_layers=[32,32,32],
    init_kernel_size=(3,3),
    bottleneck=False,
    compression_factor=1,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetBC100k12(img_dim, nb_classes, growth_rate=12, nb_filters=16, 
    nb_layers=[32,32,32],
    init_kernel_size=(3,3),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetBC250k24(img_dim, nb_classes, growth_rate=24, nb_filters=16, 
    nb_layers=[82,82,82],
    init_kernel_size=(3,3),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetBC100k40(img_dim, nb_classes, growth_rate=40, nb_filters=16, 
    nb_layers=[62,62,62],
    init_kernel_size=(3,3),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):
    
    model = DenseNet(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model


# ImageNet
def DenseNetIN121(img_dim,nb_classes,
    growth_rate=32,
    nb_filters=64,
    nb_layers=[6,12,24,16],
    init_kernel_size=(7,7),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):

    model = DenseNetIN(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetIN169(img_dim,nb_classes,
    growth_rate=32,
    nb_filters=64,
    nb_layers=[6,12,32,32],
    init_kernel_size=(7,7),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):

    model = DenseNetIN(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetIN201(img_dim,nb_classes,
    growth_rate=32,
    nb_filters=64,
    nb_layers=[6,12,48,32],
    init_kernel_size=(7,7),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):

    model = DenseNetIN(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model

def DenseNetIN264(img_dim,nb_classes,
    growth_rate=32,
    nb_filters=64,
    nb_layers=[6,12,64,48],
    init_kernel_size=(7,7),
    bottleneck=True,
    compression_factor=0.5,
    weight_decay=1E-4):

    model = DenseNetIN(img_dim,growth_rate,nb_classes,nb_filters,
                    nb_layers, init_kernel_size, bottleneck, compression_factor,
                    weight_decay)
    return model