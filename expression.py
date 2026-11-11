"""The recovered capstone baseline, with consistent inference preprocessing."""
from pathlib import Path
import cv2
import numpy as np

LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def default_model_path(root):
    rgb=Path(root)/'models/expression-rgb.keras'
    return rgb if rgb.exists() and rgb.with_suffix('.json').exists() else Path(root)/'models/expression-baseline.h5'


def preprocess(gray_face):
    if gray_face is None or gray_face.size == 0 or gray_face.ndim != 2:
        raise ValueError('Expected a nonempty grayscale face crop')
    face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
    return (face.astype(np.float32) / 255.0)[None, :, :, None]


def preprocess_frame(frame,input_shape):
    """One preprocessing path for inference, maps, and explanations (BGR in)."""
    if frame is None or frame.size==0:raise ValueError('Expected a nonempty image')
    height,width,channels=tuple(input_shape)[-3:]
    if (height,width,channels)==(48,48,1):
        return preprocess(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame)
    if (height,width,channels)!=(224,224,3):raise ValueError('Unsupported expression input shape')
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) if frame.ndim==3 else cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    interpolation=cv2.INTER_AREA if min(rgb.shape[:2])>=height else cv2.INTER_CUBIC
    return (cv2.resize(rgb,(width,height),interpolation=interpolation).astype(np.float32)/255.)[None]


def face_crop(frame,box,margin=0.):
    x,y,w,h=map(int,box);padding=round(float(margin)*max(w,h))
    return frame[max(0,y-padding):min(frame.shape[0],y+h+padding),max(0,x-padding):min(frame.shape[1],x+w+padding)]


def describe(probabilities, threshold):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected seven finite nonnegative model probabilities')
    if not np.isclose(values.sum(), 1, atol=1e-5):
        raise ValueError('Model probabilities do not sum to one')
    index = int(values.argmax())
    return {'label': LABELS[index] if values[index] >= threshold else 'uncertain',
            'top_class': LABELS[index], 'score': float(values[index]),
            'probabilities': dict(zip(LABELS, map(float, values)))}


def build_model():
    from backend import select_backend
    select_backend()
    from keras import Sequential
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
    model = Sequential([Input(shape=(48, 48, 1))])
    for filters, kernel in ((64, 5), (128, 5), (256, 3)):
        for _ in range(2):
            model.add(Conv2D(filters, kernel, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return model


class ExpressionEstimator:
    def __init__(self, model_path: Path, threshold=0.6):
        from backend import select_backend,device_info
        select_backend()
        device_info()
        if not 0 <= threshold <= 1:
            raise ValueError('Expression threshold must be between 0 and 1')
        self.threshold = threshold
        self.metadata={};self.crop_margin=0.
        model_path=Path(model_path)
        if model_path.suffix=='.keras':
            import json
            import hashlib
            from keras.models import load_model
            metadata=json.loads(model_path.with_suffix('.json').read_text(encoding='utf-8'))
            self.metadata=metadata
            if metadata.get('labels')!=list(LABELS) or metadata.get('normalization')!='divide_by_255':
                raise ValueError('Model metadata must declare the supported labels and divide_by_255 normalization')
            if metadata.get('checkpoint_sha256')!=hashlib.sha256(model_path.read_bytes()).hexdigest():
                raise ValueError('Model checkpoint does not match its metadata checksum')
            self.model=load_model(model_path,compile=False,safe_mode=True)
            shape=tuple(self.model.input_shape[1:])
            if shape not in ((48,48,1),(224,224,3)) or self.model.output_shape[-1]!=7:
                raise ValueError('Expected 48x48 grayscale or 224x224 RGB input with seven outputs')
            if shape[-1]==3:
                if metadata.get('color_space')!='RGB' or metadata.get('input_shape')!=list(shape):
                    raise ValueError('RGB model must declare its input shape and RGB color order')
                self.crop_margin=float(metadata.get('crop_margin',0.))
                if not 0<=self.crop_margin<=.5:raise ValueError('Invalid model crop margin')
        else:
            self.model = build_model()
            self.model.load_weights(str(model_path))
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.detector.empty():
            raise RuntimeError('OpenCV face detector failed to load')

    def detect(self, frame, predict=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
        if len(boxes) == 0:
            return []
        if not predict:
            return [dict(box=[int(v) for v in box]) for box in boxes]
        batch = np.concatenate([preprocess_frame(face_crop(frame,box,self.crop_margin),self.model.input_shape) for box in boxes])
        from backend import predict_numpy
        predictions = predict_numpy(self.model,batch)
        return [dict(box=[int(v) for v in box], **describe(pred, self.threshold))
                for box, pred in zip(boxes, predictions)]
