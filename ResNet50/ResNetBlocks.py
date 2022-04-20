from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import he_uniform
from keras.optimizers import adam_v2 

def identity_block(X, filters, stage, count, block_size, strides):

  f1, f2, f3 = filters
  s1, s2, s3 = strides
  X_shortcut = X

  for i in range(block_size):
    # First Conv
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides = (s1, s1), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count}")(X)
    X = BatchNormalization(axis=3, name= f"conv{stage}_B{count}")(X)
    X = Activation("relu")(X)

    # Second Conv
    X = Conv2D(filters=f2, kernel_size=(3, 3), strides = (s2, s2), padding="same", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count+1}")(X)
    X = BatchNormalization(axis=3, name= f"conv{stage}_B{count+1}")(X)
    X = Activation("relu")(X)

    # Third Conv 
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides = (s3, s3), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_{count+2}")(X)
    X = BatchNormalization(axis=3, name= f"conv{stage}_B{count+2}")(X)

    # Shorcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, filters, stage, count, strides):
  
  f1, f2, f3 = filters
  s1, s2, s3 = strides
  X_shortcut = X

  # First Conv
  X = Conv2D(filters=f1, kernel_size=(1, 1), strides = (s1, s1), kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+1}")(X)
  X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count}")(X)
  X = Activation("relu")(X)

  # Second Conv
  X = Conv2D(filters=f2, kernel_size=(3, 3), strides = (s2, s2), padding="same", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+2}")(X)
  X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count+1}")(X)
  X = Activation("relu")(X)

  # Third Conv 
  X = Conv2D(filters=f3, kernel_size=(1, 1), strides = (s3, s3), padding="valid", kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D{count+3}")(X)
  X = BatchNormalization(axis=3, name= f"conv{stage}_BC2D{count+2}")(X)

  # Shorcut
  X_shortcut = Conv2D(filters=f3, kernel_size = (1, 1), strides=(s1, s1), kernel_initializer = he_uniform(seed=0), name = f"conv{stage}_C2D")(X_shortcut)
  X_shortcut = BatchNormalization(axis=3)(X_shortcut)
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X
