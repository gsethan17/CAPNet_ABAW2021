'''
Source code for
"Causal affect prediction model using a facial image sequence" and
submissions of the Affective Behavior Analysis in-the-wild (ABAW) Competition.

Please refer to following url for the details.
https://arxiv.org/abs/2107.03886
https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/
'''

from utils import get_model, read_pickle, read_csv, load_image
import tensorflow as tf
import os
import argparse
import configparser
import cv2
import glob

# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='test',
                    help='Enter the desired type, val or test')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('./config.ini')

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

## path setting
PATH_DATA = config['PATH']['PATH_DATA']

if MODEL_KEY == 'CAPNet' :
    PATH_WEIGHT = os.path.join(config['PATH']['PATH_WEIGHT'], MODEL_KEY + '_' + str(WINDOW_SIZE), 'best_weights')
else:
    PATH_WEIGHT = os.path.join(config['PATH']['PATH_WEIGHT'], MODEL_KEY, 'best_weights')



### Model load to global variable
if MODEL_KEY == 'CAPNet' :
    MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED,
                      weight_path=PATH_WEIGHT,
                      input_size = INPUT_IMAGE_SIZE,
                      num_seq_image = NUM_SEQ_IMAGE)
else :
    MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED,
                      weight_path=PATH_WEIGHT,
                      input_size=INPUT_IMAGE_SIZE,)

## evaluation setting
BATCH_SIZE = int(config['TRAIN']['BATCH_SIZE'])


def get_postprocessing(name, img_name, pp, prior_valence, prior_arousal) :
    if name in pp['keep'] :
        if prior_valence == -10 :
            return -1
        else :
            return prior_valence, prior_arousal

    elif name in pp['zero'] :
        return 0.0, 0.0

    elif name in pp['m5'] :
        return -5.0, -5.0

    elif name in pp['both'].keys() :
        if img_name in pp['both'][name] :
            return 0.0, 0.0
        else :
            if prior_valence == -10:
                return -1
            else:
                return prior_valence, prior_arousal

    else :
        return -1



def write_submit(type='test') :
    base_dir = os.path.join(PATH_DATA, '{}_images_for_demo'.format(type))
    if not os.path.isdir(base_dir):
        print("You need the image, please download the 'test_images_for_demo'.")
        return -1

    list_tests = read_csv(os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type)))

    # post-processing
    post_dir = os.path.join(base_dir, 'post_processing_pickles')
    pp_lists = ['keep', 'zero', 'both', 'm5']
    pp_files = ['keep_past_value.pickle', 'values_to_0.pickle',
                'values_both_0_and_keep.pickle', 'values_to_m5.pickle']
    pp = {}

    for pp_name, pp_file in zip(pp_lists, pp_files) :
        if os.path.isfile(os.path.join(post_dir, pp_file)) :
            pp[pp_name] = read_pickle(os.path.join(post_dir, pp_file))
        else :
            pp[pp_name] = []
            print("There is no file : ", pp_file)


    # SAVE PATH setting
    if MODEL_KEY == 'CAPNet' :
        SAVE_PATH = os.path.join(os.getcwd(), 'results', 'VA-Set', '{}-Set'.format(type.capitalize()), MODEL_KEY + '_' + str(WINDOW_SIZE))
    else :
        SAVE_PATH = os.path.join(os.getcwd(), 'results', 'VA-Set', '{}-Set'.format(type.capitalize()), MODEL_KEY)

    if not os.path.isdir(SAVE_PATH) :
        os.makedirs(SAVE_PATH)

    for i, name in enumerate(list_tests):
        save_file_path = os.path.join(SAVE_PATH, name + ".txt")

        if "_" in name:
            if name.split('_')[-1] == 'right' or name.split('_')[-1] == 'left':
                video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(name.split('_')[:-1]) + '.*')
            else:
                video_pos = os.path.join(PATH_DATA, 'videos', name + '.*')
        else:
            video_pos = os.path.join(PATH_DATA, 'videos', name + '.*')

        if not len(glob.glob(video_pos)) == 1:
            print("Video path is not vaild : {}".format(name))
            return -1

        video_path = glob.glob(video_pos)[0]

        # count total number of frame
        capture = cv2.VideoCapture(video_path)
        total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        f = open(save_file_path, "w")
        content = "valence,arousal\n"
        f.write(content)

        count = 0
        valence, arousal = -10, -10
        for j in range(int(total_len)):
            print("{:>5} / {:>5} || {:>5} / {:>5}".format(i + 1, len(list_tests), j, int(total_len)), end='\r')

            image_path = os.path.join(base_dir, 'cropped', name, '{:0>5}'.format(j+1) + '.jpg')
            if not os.path.isfile(image_path) :
                if count == 0 :
                    valence, arousal = get_postprocessing(name, '{:0>5}'.format(j+1) + '.jpg', pp, valence, arousal)

                    content = "{},{}\n".format(valence, arousal)
                    f.write(content)

                else :
                    predicts = MODEL(xs)

                    for p in range(len(predicts)):
                        valence = predicts[p][0]
                        arousal = predicts[p][1]

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    valence, arousal = get_postprocessing(name, '{:0>5}'.format(j + 1) + '.jpg', pp, valence, arousal)

                    content = "{},{}\n".format(valence, arousal)
                    f.write(content)


                    count = 0

            else :
                x = load_image(image_path, INPUT_IMAGE_SIZE)
                x = tf.expand_dims(x, axis = 0)

                if count == 0 :
                    xs = x
                    count += 1
                else :
                    xs = tf.concat([xs, x], axis = 0)
                    count += 1

                if len(xs) < BATCH_SIZE :
                    if j == (int(total_len) - 1) :
                        predicts = MODEL(xs)

                        for p in range(len(predicts)):
                            valence = predicts[p][0]
                            arousal = predicts[p][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        count = 0

                    else :
                        continue

                else :
                    predicts = MODEL(xs)

                    for p in range(len(predicts)) :
                        valence = predicts[p][0]
                        arousal = predicts[p][1]

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    count = 0

        f.close()

def write_submit_sequence(type='test') :
    file_path = os.path.join(PATH_DATA, 'va_{}_set.csv'.format(type))
    if not os.path.isfile(file_path):
        print("type variable is not valid")
        return -1

    base_dir = os.path.join(PATH_DATA, '{}_images_for_demo'.format(type), 'cropped')
    if not os.path.isdir(base_dir):
        print("You need the image, please download the 'test_images_for_demo'.")
        return -1

    # SAVE PATH setting
    if MODEL_KEY == 'CAPNet':
        SAVE_PATH = os.path.join(os.getcwd(), 'results', 'VA-Set', '{}-Set'.format(type.capitalize()),
                                 MODEL_KEY + '_' + str(WINDOW_SIZE))
    else:
        SAVE_PATH = os.path.join(os.getcwd(), 'results', 'VA-Set', '{}-Set'.format(type.capitalize()), MODEL_KEY)

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)


    # load dataset
    data = read_pickle(os.path.join(PATH_DATA, 'va_{}_seq_list.pickle'.format(type)))

    list_tests = read_csv(file_path)


    for i, name in enumerate(list_tests):
        save_file_path = os.path.join(SAVE_PATH, name + ".txt")

        if "_" in name:
            if name.split('_')[-1] == 'right' or name.split('_')[-1] == 'left':
                video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(name.split('_')[:-1]) + '.*')
            else:
                video_pos = os.path.join(PATH_DATA, 'videos', name + '.*')
        else:
            video_pos = os.path.join(PATH_DATA, 'videos', name + '.*')

        if not len(glob.glob(video_pos)) == 1:
            print("Video path is not vaild : {}".format(name))
            return -1

        video_path = glob.glob(video_pos)[0]

        # count total number of frame
        capture = cv2.VideoCapture(video_path)
        total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        f = open(save_file_path, "w")
        content = "valence,arousal\n"
        f.write(content)

        count = 0
        for j in range(int(total_len)):
            print("{:>5} / {:>5} || {:>5} / {:>5}".format(i + 1, len(list_tests), j, int(total_len)), end='\r')

            try :
                idx = data['i'].index([name, j])
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

                    for p in range(len(predicts)):
                        valence = predicts[p][0]
                        arousal = predicts[p][1]

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    valence = -5
                    arousal = -5

                    content = "{},{}\n".format(valence, arousal)
                    f.write(content)

                    count = 0

            else:
                x = [load_image(os.path.join(base_dir, file_name), INPUT_IMAGE_SIZE) for file_name in data['x'][idx][10 - NUM_SEQ_IMAGE:]]
                x = tf.expand_dims(x, axis=0)

                if count == 0:
                    xs = x
                    count += 1
                else:
                    xs = tf.concat([xs, x], axis=0)
                    count += 1

                if len(xs) < BATCH_SIZE:
                    if j == (int(total_len) - 1):
                        predicts = MODEL(xs)

                        for p in range(len(predicts)):
                            valence = predicts[p][0]
                            arousal = predicts[p][1]

                            content = "{},{}\n".format(valence, arousal)
                            f.write(content)

                        count = 0

                    else:
                        continue

                else:
                    predicts = MODEL(xs)

                    for p in range(len(predicts)):
                        valence = predicts[p][0]
                        arousal = predicts[p][1]

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    count = 0

        f.close()

if __name__ == "__main__" :
    if MODEL_KEY == 'FER-Tuned' :
        write_submit(type=args.type)

    elif MODEL_KEY == 'CAPNet' :
        write_submit_sequence(type=args.type)

    else :
        print('Mode parser is not valid')

