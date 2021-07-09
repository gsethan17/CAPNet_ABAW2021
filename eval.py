from utils import get_model, metric_CCC, read_pickle, Dataloader, read_csv, load_image, read_txt, CCC_score_np
import tensorflow as tf
import os
import argparse
import configparser
import cv2
import glob
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
PATH_DATA = config[args.location]['PATH_DATA']
PATH_DATA_GUIDE = config[args.location]['PATH_DATA_GUIDE']
PATH_SWITCH_INFO = config[args.location]['PATH_SWITCH_INFO']
IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_latest.pickle')

if MODEL_KEY == 'CAPNet' :
    PATH_WEIGHT = os.path.join(config[args.location]['PATH_WEIGHT'], MODEL_KEY + '_' + str(WINDOW_SIZE), 'best_weights')
else:
    PATH_WEIGHT = os.path.join(config[args.location]['PATH_WEIGHT'], MODEL_KEY, 'best_weights')



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

def get_postprocessing(name, img_name, pp, prior_valence, prior_arousal) :
    print(prior_valence, prior_arousal)
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
            print("There is no file : ", pp_files)


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
        for i in range(int(total_len)):
            print("{:>5} / {:>5} || {:>5} / {:>5}".format(i + 1, len(list_tests), i, int(total_len)), end='\r')

            image_path = os.path.join(base_dir, 'cropped', name, '{:0>5}'.format(i+1) + '.jpg')
            if not os.path.isfile(image_path) :
                print(image_path)
                if count == 0 :
                    valence, arousal = get_postprocessing(name, '{:0>5}'.format(i+1) + '.jpg', pp, valence, arousal)

                    content = "{},{}\n".format(valence, arousal)
                    f.write(content)

                else :
                    predicts = MODEL(xs)

                    for p in range(len(predicts)):
                        valence = predicts[p][0]
                        arousal = predicts[p][1]

                        content = "{},{}\n".format(valence, arousal)
                        f.write(content)

                    valence, arousal = get_postprocessing(name, '{:0>5}'.format(i + 1) + '.jpg', pp, valence, arousal)

                    content = "{},{}\n".format(valence, arousal)
                    f.write(content)


                    count = 0

            else :
                print(image_path)
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
                    if i == (int(total_len) - 1):
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
    if MODEL_KEY == 'FER-Tuned' :
        write_submit(type=args.type)

    elif MODEL_KEY == 'CAPNet' :
        write_sequence(type=args.type)

    else :
        print('Mode parser is not valid')

