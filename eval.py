from utils import get_model, metric_CCC, read_pickle, Dataloader, read_csv, load_image, read_txt, CCC_score_np
import tensorflow as tf
import os
import argparse
import configparser
import time
import cv2
import glob
import numpy as np

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

# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument('--location', default='205',
                    help='Enter the server environment to be trained on')
parser.add_argument('--mode', default='show',
                    help='Enter the desired mode between write and show')

args = parser.parse_args()

# args.location

config = configparser.ConfigParser()
config.read('./config.ini')

## path setting
PATH_DATA = config[args.location]['PATH_DATA']
PATH_DATA_GUIDE = config[args.location]['PATH_DATA_GUIDE']
PATH_SWITCH_INFO = config[args.location]['PATH_SWITCH_INFO']
PATH_WEIGHT = config[args.location]['PATH_WEIGHT']
IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_latest.pickle')

## input setting
ISIMAGE = config['INPUT'].getboolean('ISIMAGE')
ISAUDIO = config['INPUT'].getboolean('ISAUDIO')

FPS = int(config['INPUT']['FPS'])
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))
WINDOW_SIZE = int(config['INPUT']['WINDOW_SIZE'])
NUM_SEQ_IMAGE = int(config['INPUT']['NUM_SEQ_IMAGE'])

SR = int(config['INPUT']['SR'])
N_MELS = int(config['INPUT']['N_MELS'])
N_FFT = int(config['INPUT']['N_FFT'])
WIN_LENGTH = int(config['INPUT']['L_WIN'])
HOP_LENGTH = int(config['INPUT']['L_HOP'])
TIME_BINS = int(WINDOW_SIZE * 1000 / HOP_LENGTH) + 1

## model setting
MODEL_KEY = str(config['MODEL']['MODEL_KEY'])
PRETRAINED = config['MODEL'].getboolean('PRETRAINED')

### Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED,
                  weight_path=PATH_WEIGHT,
                  input_size = INPUT_IMAGE_SIZE,
                  mel_size = (N_MELS, TIME_BINS),
                  num_seq_image = NUM_SEQ_IMAGE)


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

        val_metric_V.append(val_temp_metric[0].numpy())
        val_metric_A.append(val_temp_metric[1].numpy())
        val_metric_C.append(tf.math.reduce_mean(val_temp_metric).numpy())
        print("{:>5} / {:>5} || {:.4f}".format(i+1, iteration, sum(val_metric_C)/len(val_metric_C)), end='\r')

    CCC_V = sum(val_metric_V) / len(val_metric_V)
    CCC_A = sum(val_metric_A) / len(val_metric_A)
    CCC_M = sum(val_metric_C) / len(val_metric_C)

    print("Evaluation result!!")
    print("The CCC value of valence is {:.4f}".format(CCC_V))
    print("The CCC value of arousal is {:.4f}".format(CCC_A))
    print("Total CCC value is {:.4f}".format(CCC_M))

def switching(name, image, switch_images, switch_subjects) :
    if name in switch_subjects.keys():
        if image in switch_images[name]:
            object = switch_subjects[name]
        else:
            object = name
    else:
        object = name

    return object

def write_sequence(type='val') :
    file_path = os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type))
    if not os.path.isfile(file_path):
        print("type variable is not valid")
        return -1

    # Save Path setting
    weights_tag = PATH_WEIGHT.split('/')[-2]
    # tm = time.localtime(time.time())
    SAVE_PATH = os.path.join(os.getcwd(),
                             'results',
                             'evaluation',
                             weights_tag,
                             'raw')

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # load dataset
    # data_path = os.path.join(PATH_DATA, 'va_{}_seq_topfull_list.pickle'.format(type))
    data = read_pickle(VAL_DATA_PATH)

    # load switching info
    # switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
    # switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

    video_list = read_csv(file_path)

    for v, video_name in enumerate(video_list):
        # flag = False

        # write weights information
        save_file_path = os.path.join(SAVE_PATH, video_name + ".txt")

        if os.path.isfile(save_file_path):
            print("{}.txt is already exist".format(video_name))
            continue

        else:

            f = open(save_file_path, "w")
            content = "valence,arousal\n"
            f.write(content)


            if "_" in video_name:
                if video_name.split('_')[-1] == 'right' or video_name.split('_')[-1] == 'left':
                    video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(video_name.split('_')[:-1]) + '.*')
                else:
                    video_pos = os.path.join(PATH_DATA, 'videos', video_name + '.*')
            else:
                video_pos = os.path.join(PATH_DATA, 'videos', video_name + '.*')

            if not len(glob.glob(video_pos)) == 1:
                print("Video path is not vaild : {}".format(video_name))
                return -1

            video_path = glob.glob(video_pos)[0]

            # count total number of frame
            capture = cv2.VideoCapture(video_path)
            total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            # idx_list = []
            # for l in range(len(data['i'])) :
            #     if data['i'][l][0] == video_name :
            #         idx_list.append(data['i'][l])

            count = 0
            for i in range(int(total_len)):
                print("{:>5} / {:>5} || {:>5} / {:>5}".format(v + 1, len(video_list), i, int(total_len)), end='\r')
                print((video_name, i), end='')

                # for d in range(len(idx_list)) :
                #     if idx_list[d][1] == i :
                #         idx = data['i'].index(idx_list[d])
                #         break
                #     else :
                #         idx = -1

                try :
                    idx = data['i'].index([video_name, i])
                except :
                    idx = -1

                if idx == -1 :

                    if count == 0 :
                        valence = -5
                        arousal = -5

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    else :
                        predicts = MODEL(xs)

                        for i in range(len(predicts)):
                            valence = predicts[i][0]
                            arousal = predicts[i][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        valence = -5
                        arousal = -5

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                        count = 0

                else:
                    x = [load_image(os.path.join(IMAGE_PATH, file_name), INPUT_IMAGE_SIZE) for file_name in data['x'][idx][10 - NUM_SEQ_IMAGE:]]
                    x = tf.expand_dims(x, axis=0)

                    if count == 0:
                        xs = x
                        count += 1
                    else:
                        xs = tf.concat([xs, x], axis=0)
                        count += 1

                    if len(xs) < BATCH_SIZE:
                        if i == (int(total_len) - 1):
                            predicts = MODEL(xs)

                            for i in range(len(predicts)):
                                valence = predicts[i][0]
                                arousal = predicts[i][1]

                                content = "{},{}\n".format(valence, arousal)
                                f.write(content)

                            count = 0

                        else:
                            continue

                    else:
                        predicts = MODEL(xs)

                        for i in range(len(predicts)):
                            valence = predicts[i][0]
                            arousal = predicts[i][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        count = 0

            f.close()

def write_txt(type='val') :
    file_path = os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type))
    if not os.path.isfile(file_path) :
        print("type variable is not valid")
        return -1

    # Save Path setting
    if args.location == 'GSLOCAL' :
        weights_tag = PATH_WEIGHT.split('\\')[-2]
    else :
        weights_tag = PATH_WEIGHT.split('/')[-2]
    # tm = time.localtime(time.time())
    SAVE_PATH = os.path.join(os.getcwd(),
                             'results',
                             'evaluation',
                             weights_tag,
                             'raw')

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # load switching info
    switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
    switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

    video_list = read_csv(file_path)
    '''
    video_list = ['5-60-1920x1080-3',
                  '5-60-1920x1080-4',
                  '8-30-1280x720',
                  '10-60-1280x720_right',
                  '12-24-1920x1080',
                  '15-24-1920x1080',
                  '16-30-1920x1080',
                  '24-30-1920x1080-1',
                  '24-30-1920x1080-2']
    '''
    # print(video_list)
    for v, video_name in enumerate(video_list) :
        # flag = False

        # write weights information
        save_file_path = os.path.join(SAVE_PATH, video_name+".txt")

        if os.path.isfile(save_file_path) :
            print("{}.txt is already exist".format(video_name))
            continue

        else :
            f = open(save_file_path, "w")
            content = "valence,arousal\n"
            f.write(content)

            if "_" in video_name :
                if video_name.split('_')[-1] == 'right' or video_name.split('_')[-1] == 'left' :
                    video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(video_name.split('_')[:-1]) + '.*')
                else :
                    video_pos = os.path.join(PATH_DATA, 'videos', video_name + '.*')
            else :
                video_pos = os.path.join(PATH_DATA, 'videos', video_name + '.*')

            # video_path = glob.glob(os.path.join(PATH_DATA, 'videos', video_name.split('_')[0] +'*'))[0]
            if not len(glob.glob(video_pos)) == 1 :
                print("Video path is not vaild : {}".format(video_name))
                return -1

            video_path = glob.glob(video_pos)[0]

            # count total number of frame
            capture = cv2.VideoCapture(video_path)
            total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)

            # load image list
            images_list = read_csv(os.path.join(PATH_DATA_GUIDE, video_name+'.csv'))
            count = 0
            for i in range(int(total_len)) :
                print("{:>5} / {:>5} || {:>5} / {:>5}".format(v + 1, len(video_list), i, int(total_len)), end='\r')
                image_name = images_list[i]

                if image_name == '' :
                    if count == 0 :
                        valence = -5
                        arousal = -5

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    else :
                        predicts = MODEL(xs)

                        for i in range(len(predicts)):
                            valence = predicts[i][0]
                            arousal = predicts[i][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        valence = -5
                        arousal = -5

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                        count = 0

                else :
                    object = switching(video_name, image_name, switch_images, switch_subjects)
                    image_path = os.path.join(IMAGE_PATH, object, image_name)
                    x = load_image(image_path, INPUT_IMAGE_SIZE)
                    x = tf.expand_dims(x, axis = 0)

                    if count == 0 :
                        xs = x
                        count += 1
                    else :
                        xs = tf.concat([xs, x], axis = 0)
                        count += 1

                    if len(xs) < BATCH_SIZE :
                        if i == (int(total_len) - 1) :
                            predicts = MODEL(xs)

                            for i in range(len(predicts)):
                                valence = predicts[i][0]
                                arousal = predicts[i][1]

                                content = "{},{}\n".format(valence, arousal)
                                f.write(content)

                            count = 0
                            # prev_val = valence
                            # prev_aro = arousal

                        else :
                            continue

                    else :
                        predicts = MODEL(xs)

                        for i in range(len(predicts)) :
                            valence = predicts[i][0]
                            arousal = predicts[i][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        count = 0

            f.close()



if __name__ == "__main__" :
    if args.mode == 'show' :
        evaluate()
    elif args.mode == 'write' :
        write_txt()
    elif args.mode == 'write_seq' :
        write_sequence()
    elif args.mode == 'write_submit' :
        write_txt(tpye='test')
    elif args.mode == 'write_seq_submit' :
        write_sequence(tpye='test')
    else :
        print('Mode parser is not valid')

    # input_ = tf.ones((1, 224, 224, 3))
    # result = MODEL(input_)
    # result = tf.squeeze(result)
    # print(result)
    # print(result[0])
    # print(result[1])

