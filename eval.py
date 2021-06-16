from utils import get_model, metric_CCC, read_pickle, Dataloader, read_csv, load_image
import tensorflow as tf
import os
import argparse
import configparser
import time
import cv2
import glob


# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument('--location', default='205',
                    help='Enter the server environment to be trained on')
parser.add_argument('--mode', default='show',
                    help='Enter the desired mode between write and show')

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
PRETRAINED = config['MODEL'].getboolean('PRETRAINED')
### Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, weight_path=PATH_WEIGHT, input_size = INPUT_IMAGE_SIZE)

## evaluation setting
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])
SHUFFLE = config['TRAIN'].getboolean('SHUFFLE')
METRIC = metric_CCC

@tf.function
def val_step(X, Y) :
    global MODEL
    global METRIC

    predictions = MODEL(X)
    metric = METRIC(predictions, Y)

    return metric

def evaluate() :

    ################### Limit GPU Memory ###################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * 9
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')
    ##########################################################

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
        print(val_temp_metric)
        print(val_temp_metric[0])
        print(tf.math.reduce_mean(val_temp_metric))

        val_metric_V.append(val_temp_metric[0])
        val_metric_A.append(val_temp_metric[1])
        val_metric_C.append(tf.math.reduce_mean(val_temp_metric))
        print("{:>5} / {:>5} || {:.4f}, {:.4f}, {:.4f}".format(i+1, iteration, val_metric_V, val_metric_A, val_metric_C), end='\r')

    CCC_V = tf.math.reduce_mean(val_metric_V).numpy()
    CCC_A = tf.math.reduce_mean(val_metric_A).numpy()
    CCC_M = tf.math.reduce_mean(val_metric_C).numpy()

    print("Evaluation result!!")
    print("The CCC value of valence is {:.4f}".format(CCC_V))
    print("The CCC value of arousal is {:.4f}".format(CCC_A))
    print("Total CCC value is {:.4f}".format(CCC_M))

def write_txt(type='val') :
    file_path = os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type))
    if not os.path.isfile(file_path) :
        print("type variable is not valid")
        return -1

    # Save Path setting
    tm = time.localtime(time.time())
    SAVE_PATH = os.path.join(os.getcwd(),
                             'results',
                             'evaluation',
                             '{}{}_{}{}_{}'.format(tm.tm_mon,
                                                   tm.tm_mday,
                                                   tm.tm_hour,
                                                   tm.tm_min,
                                                   MODEL_KEY))

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # write weights information
    f = open(os.path.join(SAVE_PATH, "Weight.txt"), "w")
    content = "Used weights : {}\n".format(PATH_WEIGHT)
    f.write(content)
    f.close()

    # video_list = read_csv(file_path)
    video_list = ['5-60-1920x1080-3',
                  '5-60-1920x1080-4',
                  '8-30-1280x720',
                  '10-60-1280x720_right',
                  '12-24-1920x1080',
                  '15-24-1920x1080',
                  '16-30-1920x1080',
                  '24-30-1920x1080-1',
                  '24-30-1920x1080-2']
    # print(video_list)
    for v, video_name in enumerate(video_list) :
        flag = False

        # write weights information
        f = open(os.path.join(SAVE_PATH, video_name+".txt"), "w")
        content = "valence,arousal\n"
        f.write(content)


        video_path = glob.glob(os.path.join(PATH_DATA, 'videos', video_name.split('_')[0] +'*'))[0]
        if len(glob.glob(os.path.join(PATH_DATA, 'videos', video_name.split('_')[0] +'*'))) > 2 :
            print("Video path is not vaild")
            return -1

        # count total number of frame
        capture = cv2.VideoCapture(video_path)
        total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # load image list
        images_list = read_csv(os.path.join(PATH_DATA_GUIDE, video_name+'.csv'))

        for i in range(int(total_len)) :
            print("{:>5} / {:>5} || {:>5} / {:>5}".format(v + 1, len(video_list), i, int(total_len)), end='\r')
            image_name = images_list[i]

            if image_name == '' :
                if not flag :
                    valence = -5
                    arousal = -5
                else :
                    valence = prev_val
                    arousal = prev_aro

            else :
                image_path = os.path.join(IMAGE_PATH, video_name, image_name)
                x = load_image(image_path, INPUT_IMAGE_SIZE)
                x = tf.expand_dims(x, axis = 0)

                predicts = MODEL(x)
                y = tf.squeeze(predicts)

                valence = y[0]
                arousal = y[1]


            content = "{},{}\n".format(valence, arousal)
            f.write(content)

            prev_val = valence
            prev_aro = arousal

        f.close()


if __name__ == "__main__" :
    if args.mode == 'show' :
        evaluate()
    elif args.mode == 'write' :
        write_txt()
    else :
        print('Mode parser is not valid')

    # input_ = tf.ones((1, 224, 224, 3))
    # result = MODEL(input_)
    # result = tf.squeeze(result)
    # print(result)
    # print(result[0])
    # print(result[1])

