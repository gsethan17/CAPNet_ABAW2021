from utils import get_model, Dataset_generator, metric_CCC, read_pickle, Dataloader, CCC
import os


PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')
IMAGE_PATH = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/images/cropped'
# IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')

MODEL_KEY = 'pretrainedFER'  # 'pretrainedFER' / 'resnet50'
PRETRAINED = True

BATCH_SIZE = 64
SHUFFLE = False

def main() :
    val_path = os.path.join(PATH_DATA, 'va_val_list.pickle')
    val_data = read_pickle(val_path)

    val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'],
                                image_path=IMAGE_PATH,
                                batch_size=BATCH_SIZE,
                                shuffle=SHUFFLE)

    # Model Loader setup
    model = get_model(key=MODEL_KEY, preTrained=PRETRAINED)


    # predict

    iteration = len(val_dataloader)

    # set the dictionary for stack result
    val_result = {}
    val_result['ccc_v'] = []
    val_result['ccc_a'] = []
    val_result['ccc_mean'] = []

    loss_function = metric_CCC

    for i in range(iteration) :
        x, y = val_dataloader[i]

        predict = model.predict(x)

        metric, loss = loss_function(predict, y)
        val_result['ccc_v'].append(loss[0])
        val_result['ccc_a'].append(loss[1])
        val_result['ccc_mean'].append(metric)

        print("{:>6} / {:>6}\t||\tCCC(val:{:8.4f}, aro:{:8.4f}, mean:{:8.4f})".format(i+1, iteration, val_result['ccc_v'][-1], val_result['ccc_a'][-1], val_result['ccc_mean'][-1]))

    print("Evaluation result!!")
    print("Total loss value is {:.4f}".format(sum(val_result['ccc_mean']) / len(val_result['ccc_mean'])))


if __name__ == "__main__" :
    main()

    

