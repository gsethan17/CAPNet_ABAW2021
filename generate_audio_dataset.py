import moviepy.editor as mp
import os
import argparse
import configparser
from utils import read_pickle, Dataloader_audio

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

PATH_VIDEO = os.path.join(PATH_DATA, 'videos')
PATH_AUDIO = os.path.join(PATH_DATA, 'audios')
if not os.path.isdir(PATH_AUDIO) :
    os.makedirs(PATH_AUDIO)

## input setting
FPS = int(config['INPUT']['FPS'])
INPUT_IMAGE_SIZE = (int(config['INPUT']['IMAGE_WIDTH']), int(config['INPUT']['IMAGE_HEIGHT']))
WINDOW_SIZE = int(config['INPUT']['WINDOW_SIZE'])

SR = int(config['INPUT']['SR'])
N_MELS = int(config['INPUT']['N_MELS'])
N_FFT = int(config['INPUT']['N_FFT'])
WIN_LENGTH = int(config['INPUT']['L_WIN'])
HOP_LENGTH = int(config['INPUT']['L_HOP'])
TIME_BINS = int(WINDOW_SIZE * 1000 / HOP_LENGTH) + 1

def generate_audio_file() :
    global PATH_VIDEO
    global PATH_AUDIO

    error_list = []

    for name_video in os.listdir(PATH_VIDEO) :
        PATH_SAVE_AUDIO = os.path.join(PATH_AUDIO, name_video.split('.')[0] + '.wav')

        if not os.path.isfile(PATH_SAVE_AUDIO) :
            if os.path.isfile(os.path.join(PATH_VIDEO, name_video)) :
                try :
                    videoclip = mp.VideoFileClip(os.path.join(PATH_VIDEO, name_video))
                    videoclip.audio.write_audiofile(PATH_SAVE_AUDIO)
                except :
                    print("{} has something wrong!!".format(name_video))
                    error_list.append(name_video)
            else :
                print("Video file for '{}' is not exist".format(name_video))

        else :
            print("Audio file for '{}' is already exist".format(name_video))

    return error_list

def get_mel_dataset() :
    TRAIN_DATA_PATH = os.path.join(PATH_DATA, 'va_train_latest.pickle')
    VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_latest.pickle')

    train_data = read_pickle(TRAIN_DATA_PATH)
    val_data = read_pickle(VAL_DATA_PATH)

    train_dataloader = Dataloader_audio(x=train_data['x'], i=train_data['i'],
                                        data_path=PATH_DATA,
                                        fps=FPS, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                        win_length=int(SR * WIN_LENGTH / 1000),
                                        hop_length=int(SR * HOP_LENGTH / 1000),
                                        window_size=WINDOW_SIZE
                                        )

    val_dataloader = Dataloader_audio(x=val_data['x'], i=val_data['i'],
                                      data_path=PATH_DATA,
                                      fps=FPS, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                      win_length=int(SR * WIN_LENGTH / 1000),
                                      hop_length=int(SR * HOP_LENGTH / 1000),
                                      window_size=WINDOW_SIZE
                                      )

    for i in range(len(train_dataloader)) :
        data = train_dataloader[i]
        print(data[0])
        print(train_data['x'][0])
        if data[0] == train_data['x'][i] :
            print(data[0])
        if i == 0 :
            break


if __name__ == "__main__" :
    # error_list = generate_audio_file()
    # print(error_list)

    get_mel_dataset()


