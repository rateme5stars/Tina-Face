from keras import layers
from keras import Model

class ResNetBase(Model):
    def __init__(self, return_intermediate=False, **kwargs):
        '''
        Parameter
        ---------
        return_intermediate: bool, 
            - true: return all blocks
            - false: return last block
        '''
        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate
        self.zeropad_1 = layers.ZeroPadding2D(3)
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, name='conv1')
        self.bn = layers.BatchNormalization(axis=3)
        self.relu = layers.ReLU(name='relu')
        self.zeropad_2 = layers.ZeroPadding2D(1)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2)

    def call(self, x):
        x = self.zeropad_1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.zeropad_2(x)
        x = self.maxpool(x)

        x2 = self.block2(x)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        if self.return_intermediate:
            return [x2, x3, x4, x5]
        else:
            return x


class BottleNeck(Model):
    def __init__(self, dim1, dim2, strides, projection=False, **kwargs):
        '''
        Parameter
        ---------
        dim1: first kernel size of bottle neck
        dim2: second kernel size of bottle neck
        strides: stride of each kernel
        projection: 
            - true: 
            - false:
        '''
        super().__init__(**kwargs)
        self.projection = projection

        self.conv1 = layers.Conv2D(dim1, strides=1, kernel_size=1, name='conv1')
        self.bn1 = layers.BatchNormalization(axis=3, name='bn1')
        self.relu1 = layers.ReLU(name='relu1')

        self.pad = layers.ZeroPadding2D(1)
        self.conv2 = layers.Conv2D(dim1, strides=strides, kernel_size=3, name='conv2')
        self.bn2 = layers.BatchNormalization(axis=3, name='bn2')
        self.relu2 = layers.ReLU(name='relu2')

        self.conv3 = layers.Conv2D(dim2, strides=1, kernel_size=1, name='conv3')
        self.bn3 = layers.BatchNormalization(axis=3, name='bn3')
        self.relu3 = layers.ReLU(name='relu3')

        self.projection_conv = layers.Conv2D(dim2, kernel_size=1, strides=strides, name="projection")
        self.projection_bn = layers.BatchNormalization(axis=3, name='projection_bn')

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pad(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.projection:
            identity = self.projection_conv(identity)
            identity = self.projection_bn(identity)

        x += identity
        x = self.relu3(x)

        return x


class Residual(Model):
    def __init__(self, num_bottlenecks, dim1, dim2, downsampling=True, **kwargs):
        super().__init__(**kwargs)
        if downsampling:
            self.bottlenecks = [BottleNeck(dim1=dim1, dim2=dim2, strides=2, projection=True)]
        else:
            self.bottlenecks = [BottleNeck(dim1=dim1, dim2=dim2, strides=1, projection=True)]

        for i in range(1, num_bottlenecks):
            self.bottlenecks.append(BottleNeck(dim1=dim1, dim2=dim2, strides=1))

    def call(self, x):
        for btn in self.bottlenecks:
            x = btn(x)
        return x


class ResNet50(ResNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block2 = Residual(num_bottlenecks=3, dim1=64, dim2=256, downsampling=False)
        self.block3 = Residual(num_bottlenecks=4, dim1=128, dim2=512)
        self.block4 = Residual(num_bottlenecks=6, dim1=256, dim2=1024)
        self.block5 = Residual(num_bottlenecks=3, dim1=512, dim2=2048)

