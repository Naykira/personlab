from keras.layers import Input
from keras.application import resnet50


def personlab(input_tensors):
    input = Input(input_tensors)
    res = resnet50.ResNet50(include_top=False,
                            weights=None,
                            input_tensor=input,
                            input_shape=(config.TAR_H, config.TAR_W, config.TAR_D),
                            pooling=None)
    res
    return res
