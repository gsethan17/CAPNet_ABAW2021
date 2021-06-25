import moviepy.editor as mp
import os
import argparse
import configparser
from utils import read_pickle, Dataloader_audio
import pickle
import numpy as np
import librosa

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

def normalize_mel(S):
    min_level_db = -100
    return np.clip((S-min_level_db)/-min_level_db,0,1)

def get_mel_dataset() :
    global PATH_AUDIO

    for name_audio in os.listdir(PATH_AUDIO):
        PATH_SAVE_AUDIO = os.path.join(PATH_AUDIO, name_audio.split('.')[0] + '_mel.pickle')

        y, sr = librosa.load(os.path.join(PATH_AUDIO, name_audio), sr = SR)
        S = librosa.feature.melspectrogram(y=y, n_mels=N_MELS, n_fft=N_FFT,
                                           win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
        db_S = librosa.power_to_db(S, ref=np.max)
        norm_log_S = normalize_mel(db_S)

        with open(PATH_SAVE_AUDIO, 'wb') as f:
            pickle.dump(norm_log_S, f)
            print(name_audio)




if __name__ == "__main__" :
    # error_list = generate_audio_file()
    # print(error_list)
    #
    get_mel_dataset()


