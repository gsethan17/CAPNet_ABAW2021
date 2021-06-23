import moviepy.editor as mp
import os

# PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
PATH_DATA = 'D:\\Git\\ABAW2021\\data\\'
PATH_VIDEO = os.path.join(PATH_DATA, 'videos')
PATH_AUDIO = os.path.join(PATH_DATA, 'audios')
if not os.path.isdir(PATH_AUDIO) :
    os.makedirs(PATH_AUDIO)

def generate_audio_file() :
    global PATH_VIDEO
    global PATH_AUDIO

    for name_video in os.listdir(PATH_VIDEO) :
        PATH_SAVE_AUDIO = os.path.join(PATH_AUDIO, name_video.split('.')[0] + '.wav')

        if not os.path.isfile(PATH_SAVE_AUDIO) :
            if os.path.isfile(os.path.join(PATH_VIDEO, name_video)) :
                try :
                    videoclip = mp.VideoFileClip(os.path.join(PATH_VIDEO, name_video))
                    videoclip.audio.write_audiofile(PATH_SAVE_AUDIO)
                except :
                    print("{} has something wrong!!".format(name_video))
            else :
                print("Video file for '{}' is not exist".format(name_video))

        else :
            print("Audio file for '{}' is already exist".format(name_video))


if __name__ == "__main__" :
    generate_audio_file()

    '''
    audiosource = os.path.join(PATH_VIDEO, '5-60-1920x1080-4.mp4')
    videosource = os.path.join(PATH_VIDEO, '5-60-1920x1080-4_predict.mp4')
    output = os.path.join(PATH_VIDEO, '5-60-1920x1080-4_predict_withsound.mp4')

    videoclip = mp.VideoFileClip(videosource)
    audioclip = mp.VideoFileClip(audiosource)
    audioclip_get = audioclip.audio

    videoclip.audio = audioclip_get

    videoclip.write_videofile(output)
    '''

