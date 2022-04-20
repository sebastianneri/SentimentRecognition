from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform, he_uniform


class ResNet:
    def __init__(self, input_shape, classes, resnet_type):        
        self.input_shape = input_shape
        self.resnet_type = resnet_type
        self.classes = classes
        self.X_input = Input(self.input_shape)        
        self.X = ZeroPadding2D((3, 3))(self.X_input)
        self.model = self.build_model()

    def first_stage(self):
        self.X = Conv2D(filters=64, kernel_size = (7, 7), strides = (2, 2), name="conv1", kernel_initializer=he_uniform(seed=0))(self.X )
        self.X = BatchNormalization(axis=3, name="conv1_B1")(self.X)
        self.X = Activation("relu")(self.X)
        return 

    def second_stage(self):
        self.X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="conv2_1")(self.X)
        self.X = self.convolutional_block(self.X, (64, 64, 256), stage=2, count=2, strides=(1, 1, 1))
        self.X = self.identity_block(self.X, (64, 64, 256), stage=2, block_size=2, count=3, strides=(1, 1, 1))
        return 
    
    def third_stage(self):
        self.X = self.convolutional_block(self.X, (128, 128, 512), stage=3, count=1, strides=(2, 1, 1))
        self.X = self.identity_block(self.X, (128, 128, 512), stage=3, block_size=3, count=2, strides=(1, 1, 1))
        return 

    def fourth_stage(self):
        self.X = self.convolutional_block(self.X, (256, 256, 1024), stage=4, count=1, strides=(2, 1, 1))
        self.X = self.identity_block(self.X, (256, 256, 1024), stage=4, block_size=5, count=2, strides=(1, 1, 1))
        return

    def fifth_stage(self):
        self.X = self.convolutional_block(self.X, (512, 512, 2048), stage=5, count=1, strides=(2, 1, 1))
        self.X = self.identity_block(self.X, (512, 512, 2048), stage=5, block_size=2, count=2, strides=(1, 1, 1))
        return
    
    def sixth_stage(self):
        self.X = AveragePooling2D(pool_size=(2, 2), name="average_pooling")(self.X)
        self.X = Flatten()(self.X)
        self.X = Dense(units = 256, activation = 'relu')(self.X)
        if self.classes > 1:
            activation = "softmax"
        else:
            activation = "sigmoid"
        self.X = Dense(self.classes, activation=activation, name= f"fc_{self.classes}", kernel_initializer = glorot_uniform(seed=0))(self.X)
        return
    
    def identity_block(self, X, filters, stage, count, block_size, strides):
        f1, f2, f3 = filters
        s1, s2, s3 = strides
        X_shortcut = X
        for i in range(block_size):
            X = Conv2D(filters=f1, kernel_size=(1, 1), strides = (s1, s1), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count}")(X)
            X = BatchNormalization(axis=3, name= f"conv{stage}_B{count}")(X)
            X = Activation("relu")(X)

            X = Conv2D(filters=f2, kernel_size=(3, 3), strides = (s2, s2), padding="same", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count+1}")(X)
            X = BatchNormalization(axis=3, name= f"conv{stage}_B{count+1}")(X)
            X = Activation("relu")(X)

            X = Conv2D(filters=f3, kernel_size=(1, 1), strides = (s3, s3), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count+2}")(X)
            X = BatchNormalization(axis=3, name= f"conv{stage}_B{count+2}")(X)

            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)

            return X

    def convolutional_block(self, X, filters, stage, count, strides):
      f1, f2, f3 = filters
      s1, s2, s3 = strides
      X_shortcut = X

      X = Conv2D(filters=f1, kernel_size=(1, 1), strides = (s1, s1), kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+1}")(X)
      X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count}")(X)
      X = Activation("relu")(X)

      X = Conv2D(filters=f2, kernel_size=(3, 3), strides = (s2, s2), padding="same", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+2}")(X)
      X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count+1}")(X)
      X = Activation("relu")(X)

      X = Conv2D(filters=f3, kernel_size=(1, 1), strides = (s3, s3), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+3}")(X)
      X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count+2}")(X)

      X_shortcut = Conv2D(filters=f3, kernel_size = (1, 1), strides=(s1, s1), kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D")(X_shortcut)
      X_shortcut = BatchNormalization(axis=3)(X_shortcut)
      X = Add()([X, X_shortcut])
      X = Activation('relu')(X)

      return X

    def build_model(self):
        self.first_stage()
        self.second_stage()
        self.third_stage()
        self.fourth_stage()
        self.fifth_stage()
        self.sixth_stage()
        return Model(inputs = self.X_input, outputs = self.X, resnet_type)

        
