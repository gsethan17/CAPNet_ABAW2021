from utils import get_model, read_pickle, get_model, loss_ccc, metric_CCC, Dataloader_sequential
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import time
import argparse
import configparser

'''
################### Limit GPU Memory ###################
gpus = tf.config.experimental.list_physical_devices('GPU')
print("########################################")
print('{} GPU(s) is(are) available'.format(len(gpus)))
print("########################################")
# set the only one GPU and memory limit
memory_limit = 1024*9
if gpus :
    try :
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = memory_limit)])
        print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
    except RuntimeError as e :
        print(e)
else :
    print('GPU is not available')
##########################################################
'''

# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument('--location', default='205',
                    help='Enter the server environment to be trained on')
args = parser.parse_args()

args.location

config = configparser.ConfigParser()
config.read('./config.ini')

## path setting
PATH_DATA = config[args.location]['PATH_DATA']
PATH_DATA_GUIDE = config[args.location]['PATH_DATA_GUIDE']
PATH_WEIGHT = config[args.location]['PATH_WEIGHT']
IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')
TRAIN_DATA_PATH = os.path.join(PATH_DATA, 'va_train_seq_list.pickle')
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_seq_list.pickle')

## input setting
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))

## model setting
MODEL_KEY = str(config['MODEL']['MODEL_KEY'])
PRETRAINED = config['MODEL'].getboolean('PRETRAINED')
### Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, weight_path=PATH_WEIGHT, input_size = INPUT_IMAGE_SIZE)

## train setting
EPOCHS = int(config['TRAIN']['EPOCHS'])
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
SHUFFLE = config['TRAIN'].getboolean('SHUFFLE')

LEARNING_RATE = float(config['TRAIN']['LEARNING_RATE'])
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = loss_ccc
METRIC = metric_CCC

## start time setting
tm = time.localtime(time.time())

## save path setting
SAVE_PATH = os.path.join(os.getcwd(),
                         'results',
                         '{}{}_{}_{}_{}'.format(tm.tm_mon,
                                                 tm.tm_mday,
                                                 tm.tm_hour,
                                                 tm.tm_min,
                                                 MODEL_KEY))

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# write general setting
f = open(os.path.join(SAVE_PATH, "setting.txt"), "w")
setting = "Train model : {}.\nBatch size : {}.\nLearning rate : {}\n".format(MODEL_KEY, BATCH_SIZE, LEARNING_RATE)
f.write(setting)

if PRETRAINED :
    setting = "Pretrained : True\nWeight path : {}\n".format(PATH_WEIGHT)
    f.write(setting)
if SHUFFLE :
    setting = "Shuffle : True\n"
    f.write(setting)
f.close()

@tf.function
def train_step(X, Y) :
    global MODEL
    global LOSS
    global METRIC
    global OPTIMIZER

    with tf.GradientTape() as tape :
        predictions = MODEL(X)
        loss = LOSS(predictions, Y)
        metric = METRIC(predictions, Y)

    gradients = tape.gradient(loss, MODEL.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients, MODEL.trainable_variables))

    return loss, metric

@tf.function
def val_step(X, Y) :
    global MODEL
    global LOSS
    global METRIC

    predictions = MODEL(X)
    loss = LOSS(predictions, Y)
    metric = METRIC(predictions, Y)

    return loss, metric

def main() :
    train_data = read_pickle(TRAIN_DATA_PATH)
    val_data = read_pickle(VAL_DATA_PATH)
    '''
    train_dataloader = Dataloader_sequential(x=train_data['x'],
                                             y=train_data['y'],
                                             idx=train_data['i'],
                                             image_path=IMAGE_PATH,
                                             image_size=INPUT_IMAGE_SIZE,
                                             batch_size=BATCH_SIZE,
                                             shuffle=SHUFFLE)

    val_dataloader = Dataloader_sequential(x=val_data['x'],
                                             y=val_data['y'],
                                             idx=val_data['i'],
                                             image_path=IMAGE_PATH,
                                             image_size=INPUT_IMAGE_SIZE,
                                             batch_size=BATCH_SIZE,
                                             shuffle=SHUFFLE)
    '''
    # lists = []
    # for i in range(len(train_data['x'])) :
    #     count = 0
    #     for j, image_path in enumerate(train_data['x'][i]) :
    #         image = image_path.split('/')[1].split('.')[0]
    #         if image == '' :
    #             count += 1
    #
    #     if count == 9 :
    #         lists.append(i)


    # train_dataloader = Dataloader_sequential(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH,
    #                               image_size=INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # val_dataloader = Dataloader_sequential(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, image_size=INPUT_IMAGE_SIZE,
    #                             batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # print(train_dataloader[lists[0]])
    # print(val_dataloader[0])
    MODEL.build(input_shape=(1, 224, 224, 3))
    # input_ = tf.ones((1, 10, 112, 112, 3))
    #
    # output_ = MODEL.predict(input_)
    #
    # print(output_)
    print(MODEL.summary())

if __name__ == '__main__' :
    main()