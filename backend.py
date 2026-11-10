"""Select a Keras runtime before importing it; CUDA is optional, never assumed."""
import importlib.util
import os
import sys
from functools import lru_cache


def select_backend():
    if 'keras' not in sys.modules and 'KERAS_BACKEND' not in os.environ:
        os.environ['KERAS_BACKEND']='torch' if importlib.util.find_spec('torch') else 'tensorflow'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    return os.environ.get('KERAS_BACKEND','tensorflow')


@lru_cache(maxsize=1)
def device_info():
    backend=select_backend()
    if backend=='torch':
        import torch
        torch.set_num_threads(4)
        available=torch.cuda.is_available()
        # Preserve recovered-model parity; CUDA still accelerates full-float32 kernels.
        if available:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cudnn.allow_tf32=False
        return dict(backend='torch',version=torch.__version__,device=torch.cuda.get_device_name(0) if available else 'CPU',cuda=torch.version.cuda if available else None)
    import tensorflow as tf
    gpus=tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError:pass
    return dict(backend='tensorflow',version=tf.__version__,device=gpus[0].name if gpus else 'CPU',cuda=None)


def predict_numpy(model,values):
    import keras
    if keras.backend.backend()=='torch':
        import torch
        with torch.no_grad():return keras.ops.convert_to_numpy(model(values,training=False))
    return keras.ops.convert_to_numpy(model(values,training=False))
