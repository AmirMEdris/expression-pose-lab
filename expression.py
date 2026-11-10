"""The recovered capstone baseline, with consistent inference preprocessing."""
from pathlib import Path
import cv2
import numpy as np

LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def preprocess(gray_face):
    if gray_face is None or gray_face.size == 0 or gray_face.ndim != 2:
        raise ValueError('Expected a nonempty grayscale face crop')
    face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
    return (face.astype(np.float32) / 255.0)[None, :, :, None]


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
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
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
        if not 0 <= threshold <= 1:
            raise ValueError('Expression threshold must be between 0 and 1')
        self.threshold = threshold
        self.model = build_model()
        self.model.load_weights(str(model_path))
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.detector.empty():
            raise RuntimeError('OpenCV face detector failed to load')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
        if len(boxes) == 0:
            return []
        batch = np.concatenate([preprocess(gray[y:y+h, x:x+w]) for x,y,w,h in boxes])
        predictions = self.model(batch, training=False).numpy()
        return [dict(box=[int(v) for v in box], **describe(pred, self.threshold))
                for box, pred in zip(boxes, predictions)]
