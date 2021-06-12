import moviepy.editor as mp
import os

PATH_VIDEO = os.path.join(os.getcwd(), 'data', 'videos')
PATH_AUDIO = os.path.join(os.getcwd(), 'data', 'audios')
if not os.path.isdir(PATH_AUDIO) :
    os.makedirs(PATH_AUDIO)

def generate_audio_file() :
    global PATH_VIDEO
    global PATH_AUDIO

    for name_video in os.listdir(PATH_VIDEO) :
        PATH_SAVE_AUDIO = os.path.join(PATH_AUDIO, name_video.split('.')[0] + '.mp3')

        if not os.path.isfile(PATH_SAVE_AUDIO) :
            videoclip = mp.VideoFileClip(os.path.join(PATH_VIDEO, name_video))
            videoclip.audio.write_audiofile(PATH_SAVE_AUDIO)

        else :
            print("Audio file for '{}' is already exist".format(name_video))


if __name__ == "__main__" :
    generate_audio_file()

