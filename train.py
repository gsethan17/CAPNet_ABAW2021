'''
Source code for
"Causal affect prediction model using a facial image sequence" and
submissions of the Affective Behavior Analysis in-the-wild (ABAW) Competition.

Please refer to following url for the details.
https://arxiv.org/abs/2107.03886
https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/
'''

from utils import get_model, loss_ccc, metric_CCC, read_pickle, Dataloader, Dataloader_sequential
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import time
import configparser


# Basic configuration
config = configparser.ConfigParser()
config.read('./config.ini')

## path setting
PATH_DATA = config['PATH']['PATH_DATA']
PATH_DATA_GUIDE = config['PATH']['PATH_DATA_GUIDE']
PATH_WEIGHT = config['PATH']['PATH_WEIGHT']
IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')
TRAIN_DATA_PATH = os.path.join(PATH_DATA, 'va_train_latest.pickle')
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_latest.pickle')

## model setting
MODEL_KEY = str(config['MODEL']['MODEL_KEY'])
PRETRAINED = config['MODEL'].getboolean('PRETRAINED')

## input setting
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))

if MODEL_KEY == 'CAPNet' :
    WINDOW_SIZE = int(config['INPUT']['WINDOW_SIZE'])

    FPS = 30
    STRIDE = 10

    # NUM_SEQ_IMAGE = int(config['INPUT']['NUM_SEQ_IMAGE'])
    NUM_SEQ_IMAGE = int(WINDOW_SIZE * FPS / STRIDE)

## train setting
EPOCHS = int(config['TRAIN']['EPOCHS'])
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
SHUFFLE = config['TRAIN'].getboolean('SHUFFLE')
LEARNING_RATE = float(config['TRAIN']['LEARNING_RATE'])
DROPOUT_RATE = float(config['TRAIN']['DROPOUT_RATE'])
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = loss_ccc
METRIC = metric_CCC

### Model load to global variable
if MODEL_KEY == 'CAPNet' :
    MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED,
                      weight_path=PATH_WEIGHT,
                      input_size=INPUT_IMAGE_SIZE,
                      dropout_rate=DROPOUT_RATE,
                      num_seq_image=NUM_SEQ_IMAGE)
else :
    MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED,
                      weight_path=PATH_WEIGHT,
                      input_size=INPUT_IMAGE_SIZE,)

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
setting = "Train model : {}.\nBatch size : {}.\nLearning rate : {}\nDropout : {}\n".format(MODEL_KEY, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE)
f.write(setting)
if MODEL_KEY == 'CAPNet' :
    setting = "Number of sequential images : {}\n".format(NUM_SEQ_IMAGE)
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
        predictions = MODEL(X, training=True)
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
    if MODEL_KEY == 'FER-Tuned' :
        train_dataloader = Dataloader(x=train_data['x'], y=train_data['y'],
                                      image_path=IMAGE_PATH,
                                      image_size = INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE,
                                      shuffle=SHUFFLE,)
        val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'],
                                      image_path=IMAGE_PATH,
                                      image_size=INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE,
                                      shuffle=SHUFFLE, )
    elif MODEL_KEY == 'CAPNet':
        train_dataloader = Dataloader_sequential(x=train_data['x'], y=train_data['y'], i=train_data['i'],
                                                 image_path=IMAGE_PATH,
                                                 image_size = INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                                                 num_seq_image=NUM_SEQ_IMAGE)
        val_dataloader = Dataloader_sequential(x=val_data['x'], y=val_data['y'], i=val_data['i'],
                                               image_path=IMAGE_PATH,
                                               image_size=INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                                               num_seq_image=NUM_SEQ_IMAGE)
    else :
        print("MODEL_KEY value is not valid")
        return -1

    ed_train = time.time()
    print("Dataloader has been build ({:.1f}seconds).".format(ed_train - st_build))


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
            train_loss.append(train_temp_loss.numpy())
            train_metric_V.append(train_temp_metric[0].numpy())
            train_metric_A.append(train_temp_metric[1].numpy())
            train_metric_C.append(tf.math.reduce_mean(train_temp_metric).numpy())
        train_dataloader.on_epoch_end()
        ed_train = time.time()
        print("Training is completed...", end="\r")

        results['train_loss'].append(sum(train_loss)/len(train_loss))
        results['train_ccc_V'].append(sum(train_metric_V)/len(train_metric_V))
        results['train_ccc_A'].append(sum(train_metric_A)/len(train_metric_A))
        results['train_CCC'].append(sum(train_metric_C)/len(train_metric_C))

        val_loss = []
        val_metric_V = []
        val_metric_A = []
        val_metric_C = []

        print("Validation Start...", end="\r")
        for j in range(len(val_dataloader)) :

            x_val, y_val = val_dataloader[j]
            print("Validation : {} / {}".format(j + 1, len(val_dataloader)), end="\r")
            val_temp_loss, val_temp_metric = val_step(x_val, y_val)
            val_loss.append(val_temp_loss.numpy())
            val_metric_V.append(val_temp_metric[0].numpy())
            val_metric_A.append(val_temp_metric[1].numpy())
            val_metric_C.append(tf.math.reduce_mean(val_temp_metric).numpy())
        val_dataloader.on_epoch_end()

        ed_val = time.time()

        MODEL.save_weights(os.path.join(SAVE_PATH, "{}epoch_weights".format(epoch+1)))

        if epoch > 0 :
            if (sum(val_metric_C) / len(val_metric_C)) > max(results['val_CCC']) :
                # save best weights
                MODEL.save_weights(os.path.join(SAVE_PATH, "best_weights"))

        results['val_loss'].append(sum(val_loss) / len(val_loss))
        results['val_ccc_V'].append(sum(val_metric_V) / len(val_metric_V))
        results['val_ccc_A'].append(sum(val_metric_A) / len(val_metric_A))
        results['val_CCC'].append(sum(val_metric_C) / len(val_metric_C))

        print("{:>3} / {:>3} || train_loss:{:8.4f}, train_CCC:{:8.4f}, val_loss:{:8.4f}, val_CCC:{:8.4f} || TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(epoch + 1, EPOCHS,
                                                                                      results['train_loss'][-1],
                                                                                      results['train_CCC'][-1],
                                                                                      results['val_loss'][-1],
                                                                                      results['val_CCC'][-1],
                                                                                      (ed_train - st_train),
                                                                                      (ed_val - ed_train)))

        # early stop
        if epoch > 5 :
            if results['val_CCC'][-5] > max(results['val_CCC'][-4:]) :
                break

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(SAVE_PATH, 'Results.csv'), index=False)


if __name__ == "__main__" :
    main()
