def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'mobilev2pruning':
        from .mobilenetv2pruning import MobileNetV2
        return MobileNetV2
    elif network_name == 'mobilev2':
        from .mobilenetv2 import MobileNetV2
        return MobileNetV2
    elif network_name == 'squeeze':
        from .squeezenet import SqueezeNet
        return SqueezeNet
    elif network_name == 'shuffle':
        from .shufflenet import ShuffleNet
        return ShuffleNet
    elif network_name == 'resnet50':
        from .resnet import resnet50
        return resnet50
    elif network_name == 'resnet101':
        from .resnet import resnet101
        return resnet101
    elif network_name == 'resnet152':
        from .resnet import resnet152
        return resnet152
    elif network_name == 'xception':
        from .xception import Xception
        return Xception
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
