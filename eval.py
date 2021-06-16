from utils import get_model, metric_CCC, read_pickle, Dataloader, read_csv, load_image, read_txt, CCC_score_np
import tensorflow as tf
import os
import argparse
import configparser
import time
import cv2
import glob
import numpy as np
'''
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
'''
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

def write_txt(type='val') :
    file_path = os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type))
    if not os.path.isfile(file_path) :
        print("type variable is not valid")
        return -1

    # Save Path setting
    weights_tag = PATH_WEIGHT.split('/')[-2]
    # tm = time.localtime(time.time())
    SAVE_PATH = os.path.join(os.getcwd(),
                             'results',
                             'evaluation',
                             weights_tag)

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # write weights information
    # f = open(os.path.join(SAVE_PATH, "Weight.txt"), "w")
    # content = "Used weights : {}\n".format(PATH_WEIGHT)
    # f.write(content)
    # f.close()

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
        flag = False

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
                    video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(video_name.split('_')[:-1]) + '*')
                else :
                    video_pos = os.path.join(PATH_DATA, 'videos', video_name + '*')
            else :
                video_pos = os.path.join(PATH_DATA, 'videos', video_name + '*')

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

def compare() :
    weights_tag = str(input("Please enter the folder name to compare : \n"))
    prediction_path = os.path.join(os.getcwd(), 'results', 'evaluation', weights_tag)
    prediction_lists = os.listdir(prediction_path)
    # prediction_lists.pop(prediction_lists.index('Weight.txt'))

    total_ccc_V = []
    total_ccc_A = []

    for i, prediction_list in enumerate(prediction_lists) :
        predictions = read_txt(os.path.join(prediction_path, prediction_list))

        pred = []
        for i in range(len(predictions)-1) :
            pred.append([float(x) for x in predictions[(i+1)]])

        pred = np.array(pred)

        predictions_V = pred[:, :1]
        predictions_A = pred[:, 1:]

        gts = read_txt(os.path.join(PATH_DATA, 'annotations', 'VA_Set', 'Validation_Set', prediction_list))

        gt = []
        for j in range(len(gts)-1) :
            gt.append([float(x) for x in gts[(j+1)]])
        gt = np.array(gt)

        gts_V = gt[:, :1]
        gts_A = gt[:, 1:]

        valence_ccc_score = CCC_score_np(predictions_V, gts_V[:len(predictions_V)])
        arousal_ccc_score = CCC_score_np(predictions_A, gts_A[:len(predictions_A)])

        total_ccc_V.append(valence_ccc_score)
        total_ccc_A.append(arousal_ccc_score)

        print("{} : {:.4f}, {:.4f}".format(prediction_list, valence_ccc_score, arousal_ccc_score))

    ccc_V = sum(total_ccc_V) / len(total_ccc_V)
    ccc_A = sum(total_ccc_A) / len(total_ccc_A)
    ccc_M = (ccc_V + ccc_A) / 2

    print("")
    print("Comparision result!!")
    print("The CCC value of valence is {:.4f}".format(ccc_V))
    print("The CCC value of arousal is {:.4f}".format(ccc_A))
    print("Total CCC value is {:.4f}".format(ccc_M))

if __name__ == "__main__" :
    if args.mode == 'show' :
        evaluate()
    elif args.mode == 'write' :
        write_txt()
    elif args.mode == 'compare' :
        compare()
    elif args.mode == 'write_submit' :
        write_txt(tpye='test')
    else :
        print('Mode parser is not valid')

    # input_ = tf.ones((1, 224, 224, 3))
    # result = MODEL(input_)
    # result = tf.squeeze(result)
    # print(result)
    # print(result[0])
    # print(result[1])

