from glob import glob
from models.ResNet import ResNet34


# Model Load Function
def get_model(key='FER', preTrained = True) :
    if ket == 'FER' :
        # Model load
        model = ResNet34(cardinality = 32, se = 'parallel_add')
        
        if preTrained :
            # load pre-trained weights
            weight_path = os.path.join(os.getcwd(), 'models', 'ResNeXt34_Parallel_add', 'checkpoint_4_300000-320739.ckpt')
            assert len(glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
            model.load_weights(weight_path)
        
        return model


if __name__ == '__main__' :
    model = get_model()
    print(model.summary())
