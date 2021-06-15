from utils import get_model, metric_CCC, read_pickle, Dataloader
import tensorflow as tf
import os
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
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_list.pickle')

## input setting
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))

## model setting
MODEL_KEY = str(config['MODEL']['MODEL_KEY'])
PRETRAINED = config['MODEL']['PRETRAINED']
### Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, path_weight=PATH_WEIGHT, input_size = INPUT_IMAGE_SIZE)

## evaluation setting
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
SHUFFLE = config['TRAIN']['SHUFFLE']
METRIC = metric_CCC

@tf.function
def val_step(X, Y) :
    global MODEL
    global METRIC

    predictions = MODEL(X)
    metric = METRIC(predictions, Y)

    return metric

def evaluate() :
    val_data = read_pickle(VAL_DATA_PATH)

    val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'],
                                image_path=IMAGE_PATH,
                                image_size=INPUT_IMAGE_SIZE,
                                batch_size=BATCH_SIZE,
                                shuffle=SHUFFLE)

    # predict

    iteration = len(val_dataloader)

    # set the dictionary for stack result
    val_result = {}
    val_result['ccc_v'] = []
    val_result['ccc_a'] = []
    val_result['ccc_mean'] = []

    print("Evaluation Start...")
    val_metric_V = []
    val_metric_A = []
    val_metric_C = []

    for i in range(iteration) :
        x, y = val_dataloader[i]

        val_temp_metric = val_step(x, y)

        val_metric_V.append(val_temp_metric[0])
        val_metric_A.append(val_temp_metric[1])
        val_metric_C.append(tf.math.reduce_mean(val_temp_metric))

    CCC_V = tf.math.reduce_mean(val_metric_V).numpy()
    CCC_A = tf.math.reduce_mean(val_metric_A).numpy()
    CCC_M = tf.math.reduce_mean(val_metric_C).numpy()

    print("Evaluation result!!")
    print("The CCC value of valence is {:.4f}".format(CCC_V))
    print("The CCC value of arousal is {:.4f}".format(CCC_A))
    print("Total CCC value is {:.4f}".format(CCC_M))


if __name__ == "__main__" :
    evaluate()

    

