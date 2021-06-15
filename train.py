from utils import get_model, loss_ccc, metric_CCC, read_csv, read_pickle, Dataloader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import time
import argparse
import configparser


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
TRAIN_DATA_PATH = os.path.join(PATH_DATA, 'va_train_list.pickle')
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_list.pickle')

## input setting
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))

## model setting
MODEL_KEY = str(config['MODEL']['MODEL_KEY'])
PRETRAINED = config['MODEL']['PRETRAINED']
### Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, path_weight=PATH_WEIGHT, input_size = INPUT_IMAGE_SIZE)

## train setting
EPOCHS = int(config['TRAIN']['EPOCHS'])
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
SHUFFLE = config['TRAIN']['SHUFFLE']

LEARNING_RATE = float(config['TRAIN']['LEARNING_RATE'])
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = loss_ccc
METRIC = metric_CCC

## start time setting
tm = time.localtime(time.time())

## save path setting
SAVE_PATH = os.path.join(os.getcwd(),
                         'results',
                         '{}{}_{}{}_{}'.format(tm.tm_mon,
                                                 tm.tm_mday,
                                                 tm.tm_hour,
                                                 tm.tm_min,
                                                 MODEL_KEY))

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# write general setting
f = open(os.path.join(SAVE_PATH, "setting.txt"), "w")
setting = "Train model : {}.\nBatch size : {}.\nLearning late : {}\n".format(MODEL_KEY, BATCH_SIZE, LEARNING_RATE)
f.write(setting)
if PRETRAINED :
    setting = "Pretrained : True\n"
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

    print("Build the data loader")
    st_build = time.time()
    train_dataloader = Dataloader(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH, image_size = INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ed_train = time.time()
    print("Train data has been build ({:.1f}seconds).".format(ed_train - st_build))

    val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, image_size = INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ed_val = time.time()
    print("Validation data has been build ({:.1f}seconds).".format(ed_val - ed_train))


    ## use gradient tape
    results = {}
    results['train_loss'] = []
    results['train_ccc_V'] = []
    results['train_ccc_A'] = []
    results['train_CCC'] = []

    results['val_loss'] = []
    results['val_ccc_V'] = []
    results['val_ccc_A'] = []
    results['val_CCC'] = []

    print("Training Start...")
    for epoch in range(EPOCHS) :
        train_loss = []
        train_metric_V = []
        train_metric_A = []
        train_metric_C = []

        st_train = time.time()
        for i in range(len(train_dataloader)) :

            x_train, y_train = train_dataloader[i]
            print("Training : {} / {}".format(i + 1, len(train_dataloader)), end="\r")
            train_temp_loss, train_temp_metric = train_step(x_train, y_train)
            train_loss.append(train_temp_loss)
            train_metric_V.append(train_temp_metric[0])
            train_metric_A.append(train_temp_metric[1])
            train_metric_C.append(tf.math.reduce_mean(train_temp_metric))
        ed_train = time.time()
        print("Training is completed...", end="\r")

        results['train_loss'].append(tf.math.reduce_mean(train_loss).numpy())
        results['train_ccc_V'].append(tf.math.reduce_mean(train_metric_V).numpy())
        results['train_ccc_A'].append(tf.math.reduce_mean(train_metric_A).numpy())
        results['train_CCC'].append(tf.math.reduce_mean(train_metric_C).numpy())

        val_loss = []
        val_metric_V = []
        val_metric_A = []
        val_metric_C = []

        print("Validation Start...", end="\r")
        for j in range(len(val_dataloader)) :

            x_val, y_val = val_dataloader[j]
            print("Validation : {} / {}".format(j + 1, len(val_dataloader)), end="\r")
            val_temp_loss, val_temp_metric = val_step(x_val, y_val)
            val_loss.append(val_temp_loss)
            val_metric_V.append(val_temp_metric[0])
            val_metric_A.append(val_temp_metric[1])
            val_metric_C.append(tf.math.reduce_mean(val_temp_metric))

        ed_val = time.time()

        MODEL.save_weights(os.path.join(SAVE_PATH, "{}epoch_weights".format(epoch+1)))

        if tf.math.reduce_mean(val_metric_C) > tf.math.reduce_max(results['val_CCC']) :
            # save best weights
            MODEL.save_weights(os.path.join(SAVE_PATH, "best_weights"))

        results['val_loss'].append(tf.math.reduce_mean(val_loss).numpy())
        results['val_ccc_V'].append(tf.math.reduce_mean(val_metric_V).numpy())
        results['val_ccc_A'].append(tf.math.reduce_mean(val_metric_A).numpy())
        results['val_CCC'].append(tf.math.reduce_mean(val_metric_C).numpy())
        
        print("{:>3} / {:>3} || train_loss:{:8.4f}, train_CCC:{:8.4f}, val_loss:{:8.4f}, val_CCC:{:8.4f} || TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(epoch + 1, EPOCHS,
                                                                                      results['train_loss'][-1],
                                                                                      results['train_CCC'][-1],
                                                                                      results['val_loss'][-1],
                                                                                      results['val_CCC'][-1],
                                                                                      (ed_train - st_train),
                                                                                      (ed_val - ed_train)))

        # early stop
        if epoch > 5 :
            if results['val_CCC'][-5] > tf.math.reduce_max(results['val_CCC'][-4:]) :
                break

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(SAVE_PATH, 'Results.csv'), index=False)


if __name__ == "__main__" :
    main()
