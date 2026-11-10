"""Filter Explorer uses one-based convolution layers and zero-based filter IDs."""


def resolve_filter(catalog,layer_number,channel):
    if isinstance(layer_number,bool) or not isinstance(layer_number,int) or not 1<=layer_number<=len(catalog):
        raise ValueError('Choose a valid convolution layer')
    selected=catalog[layer_number-1]
    if isinstance(channel,bool) or not isinstance(channel,int) or not 0<=channel<selected['channels']:
        raise ValueError('Choose a filter available in this layer')
    return selected['name'],channel


def filter_activation(activation,channel):
    if isinstance(channel,bool) or not isinstance(channel,int) or not 0<=channel<int(activation.shape[-1]):
        raise ValueError('Filter ID is outside the selected layer')
    return activation[...,channel:channel+1]
