Model API

Layers API

Callbacks API

Optimizers

Metrics

Losses

Data loading

Built-in small datasets

Keras Applications 

Mixed precision

Utilities

KerasTuner

-------

The Model class

The Sequential class

Model training APIs

Model saving & serialization APIs

-------

The base Layer class

Layer activations

Layer weight initializers

Layer weight regularizers

Layer weight constraints 

Core layers

Convolution layers

Pooling layers

Recurrent layers

Preprocessing layers

Normalization layers 

Regularization layers 

Attention layers

Reshaping layers 

Merging layers 

Locally-connected layers 

Activation layers

-------

Base Callback class

ModelCheckpoint

TensorBoard

EarlyStopping

LearningRateScheduler

ReduceLROnPlateau

RemoteMonitor

LambdaCallback

TerminateOnNaN

CSVLogger

ProgbarLogger

BackupAndRestore 

-------

SGD

RMSprop

Adam

Adadelta

Adagrad

Adamax

Nadam

Ftrl

-------

Accuracy metrics

Probabilistic metrics

Regression metrics

Classification metrics based on True/False positives & negatives 

Image segmentation metrics 

Hinge metrics for "maximum-margin" classification

-------

Probabilistic losses

Regression losses

Hinge losses for "maximum-margin" classification

--------

Image data loading

Timeseries data loading

Text data loading

-----------

MNIST digits classification dataset 

CIFAR10 small images classification dataset

CIFAR100 small images classification dataset

IMDB movie review sentiment classification dataset 

Fashion MNIST dataset, an alternative to MNIST

Boston Housing price regression dataset

------

Xception

EfficientNet B0 to B7

EfficientNetV2 B0 t0 B3 and S, M, L

VGG16 and VGG19

ResNet and ResNetV2

MobileNet, MobileNetV2, and MobileNetV3 

DenseNet

NasNetLarge and NasNetMobile

InceptionV3

InceptionResNetV2

---------------

Mixed precision policy API

LossScaleOptimizer

------

Model plotting utilities

Serialization utilities

Python & Numpy utilities

Backend utilities

-------

HyperParameters

Tuners 

Oracles

HyperModels

-------------------

Model class

summary method

get_layer method

----------

Sequential class

add method

pop method

-------

compile method

fit method

evaluate method

predict method

train_on_batch method

test_on_batch method

predict_on_batch method

run_eagerly property

-------

save method

save_model function

load_model function 

get_weights method 

set_weights method 

save_weights method 

load_weights method 

get_config method 

from_config method 

model_from_config function 

to_json method 

model_from_json function 

clone_model function

------

Layer class

weights property

trainable_weights property

non_trainable_weights property

trainable property

get_weights method 

set_weights method 

get_config method 

add_loss method 

add_metric method 

losses property 

metrics property 

dynamic property 

-----

relu function 

sigmoid function 

softmax function 

softplus function 

softsign function 

tanh function 

selu function 

elu function 

exponential function 

-----

RandomNormal class

RandomUniform class

TruncatedNormal class 

Zeros class

Ones class 

GlorotNormal class 

GlorotUniform class 

HeNormal class 

HeUniform class 

Identity class 

Orthogonal class 

Constant class 

VarianceScaling class 

-------

L1 class

L2 class 

L1L2 class 


-----

MaxNorm class 

MinMaxNorm class 

NonNeg class 

UnitNorm class 

RadialConstraint class 

-------

Input object

Dense layer

Activation layer

Embedding layer

Masking layer 

Lambda layer

------

Conv1D layer

Conv2D layer 

Conv3D layer

SeparableConv1D layer 

SeparableConv2D layer 

DepthwiseConv2D layer 

Conv2DTranspose layer 

Conv3DTranspose layer

-----

MaxPooling1D layer

MaxPooling2D layer 

MaxPooling3D layer

AveragePooling1D layer

AveragePooling2D layer 

AveragePooling3D layer 

GlobalMaxPooling1D layer

GlobalMaxPooling2D layer 

GlobalMaxPooling3D layer 

GlobalAveragePooling1D layer 

GlobalAveragePooling2D layer

GlobalAveragePooling3D layer

------

LSTM layer

GRU layer 

SimpleRNN layer

TimeDistributed layer

Bidirectional layer

ConvLSTM1D layer - 

ConvLSTM2D layer - 

ConvLSTM3D layer - 

BaseRNN layer 

-----

Text preprocessing

Numerical features preprocessing layers

Categorical features preprocessing layers

Image preprocessing layers

Image augmentation layers

-----

BatchNormalization layer

LayerNormalization layer

-----

Dropout layer 

SpatialDropout1D layer
 
SpatialDropout2D layer 

SpatialDropout3D layer 

GaussianDropout layer 

GaussianNoise layer 

ActivityRegularization layer 

AlphaDropout layer

----

MultiHeadAttention layer

Attention layer 

AdditiveAttention layer 

-----

Reshape layer 

Flatten layer

RepeatVector layer 

Permute layer 

Cropping1D layer 

Cropping2D layer 

Cropping3D layer 

UpSampling1D layer 

UpSampling2D layer 

UpSampling3D layer 

ZeroPadding1D layer

ZeroPadding2D layer

ZeroPadding3D layer 

-------

Concatenate layer

Average layer

Maximum layer

Minimum layer 

Add layer 

Subtract layer 

Multiply layer 

Dot layer 

-----

LocallyConnected1D layer 

LocallyConnected2D layer

-----

ReLU layer 

Softmax layer

LeakyReLU layer 

PReLU layer 

ELU layer 

ThresholdedReLU layer 

----

TextVectorization layer

----

Normalization layer

Discretization layer

-----

CategoryEncoding layer

Hashing layer

StringLookup layer

IntegerLookup layer

-----

Resizing layer

Rescaling layer

CenterCrop layer

------

RandomCrop layer

RandomFlip layer

RandomTranslation layer

RandomRotation layer

RandomZoom layer

RandomHeight layer

RandomWidth layer

RandomContrast layer 

----

Accuracy class

BinaryAccuracy class

CategoricalAccuracy class

SparseCategoricalAccuracy class

TopKCategoricalAccuracy class

SparseTopKCategoricalAccuracy class 

-----

BinaryCrossentropy class

CategoricalCrossentropy class

SparseCategoricalCrossentropy class 

KLDivergence class

Poisson class

------

MeanSquaredError class 

RootMeanSquaredError class 

MeanAbsoluteError class 

MeanAbsolutePercentageError class 

MeanSquaredLogarithmicError class 

CosineSimilarity class 

LogCoshError class 

-----

AUC class

Precision class 

Recall class 

TruePositives class

TrueNegatives class 

FalsePositives class

FalseNegatives class 

PrecisionAtRecall class 

SensitivityAtSpecificity class 

SpecificityAtSensitivity class 

----

MeanIoU class

----

Hinge class 

SquaredHinge class

CategoricalHinge class 

----

BinaryCrossentropy class

CategoricalCrossentropy class

SparseCategoricalCrossentropy class

Poisson class 

binary_crossentropy function

categorical_crossentropy function 

sparse_categorical_crossentropy function 

possion function

KLDivergence class 

kl_divergence function

-----

MeanSquaredError class 

MeanAbsoluteError class

MeanAbsolutePercentageError class 

MeanSquaredLogarithmicError class 

CosineSimilarity class 

mean_squared_error function

mean_absolute_error function 

mean_absolute_percentage_error function

mean_absolute_logarithmic_error function

cosine_similarity function

Huber class

huber function

LogCosh class 

log_cosh function 

-------------

Hinge class 

SquaredHinge class 

CategoricalHinge class 

hinge function

squared_hinge function 

categorical_hinge function

-------

image_dataset_from_directory function

load_img function

img_to_array function

save_img function 

-----

timeseries_dataset_from_array function

------

text_dataset_from_directory function

--------

load_data function

------

get_word_index function

---

Xception

VGG16

VGG19

ResNet50

ResNet50V2

ResNet101

ResNet101V2

ResNet152

ResNet152V2

InceptionV3

InceptionResNetV2

MobileNet

MobileNetV2

DenseNet121

DenseNet169

DenseNet201

NasNetMobile

NasNetLarge

EfficientNetB0

EfficientNetB1

EfficientNetB2

EfficientNetB3

EfficientNetB4

EfficientNetB5

EfficientNetB6

EfficientNetB7

EfficientNetV2B0

EfficientNetV2B1

EfficientNetV2B2

EfficientNetV2B3

EfficientNetV2S

EfficientNetV2M

EfficientNetV2L

------

Policy class

global_policy function

set_global_policy function

------

LossScaleOptimizer class

-------

plot_model function

model_to_dot function

-------

custom_object_scope class

get_custom_objects function

register_keras_serializable function

serialize_keras_object function

deserialize_keras_object function

-------

set_random_seed function

to_categorical function

normalize function

get_file function 

Progbar class 

Sequence class 

-------

clear_session function

floatx function

set_floatx function 

image_data_format function

set_image_data_format function

epsilon function

set_epsilon function 

is_keras_tensor function

get_uid function 

rnn function

------

HyperParameter class

Boolean method

Choice method 

Fixed method 

Float method 

Int method 

condition_scope method 

get method 

------------

The base Tuner class 

RandomSearch Tuner 

BayesianOptimization Tuner 

Hyperband Tuner 

Sklearn Tuner 

---------

The base Oracle class 

RandomSearch Oracle

BayesianOptimization Oracle 

Hyperband Oracle 

-----

The base HyperModel class 

HyperResNet

HyperXception

------
































































