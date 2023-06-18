from models.cnn_mnn import CNN
from models.logistic_regression import LR


def create_model_instance(model_type):
    if model_type == 'cnn_mnn':
        return CNN()
    elif model_type == 'lr_mnn':
        input_size = 8776
        return LR(input_size=input_size)
    else:
        raise ValueError("Not valid model type")
