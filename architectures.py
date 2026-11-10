"""Comparable 48x48 expression classifiers; no network or training at import."""
def compact_model(spatial=False):
    from backend import select_backend
    select_backend()
    import keras as k
    layers=k.layers
    inputs=k.Input((48,48,1))
    x=inputs
    for filters in (32,64,128):
        for _ in range(2):
            x=layers.Conv2D(filters,3,padding='same',use_bias=False)(x)
            x=layers.BatchNormalization()(x)
            x=layers.Activation('relu')(x)
        x=layers.MaxPooling2D()(x)
        x=layers.Dropout(.2)(x)
    x=layers.Flatten()(x) if spatial else layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(.4)(x)
    return k.Model(inputs,layers.Dense(7,activation='softmax')(x),name='expression_spatial' if spatial else 'expression_compact')


def original_final_model():
    """Reconstruct RevisedMainNotebook cell 21; weights are not recovered."""
    from backend import select_backend
    select_backend()
    from keras import Sequential,layers as L
    m=Sequential([L.Input((48,48,1))],name='original_final')
    for i,(filters,kernel) in enumerate(((256,2),(256,2),(256,5),(512,2),(512,4),(512,2))):
        m.add(L.Conv2D(filters,kernel,activation='relu',padding='same'))
        if i<2:
            m.add(L.BatchNormalization());m.add(L.Dropout(.2))
        else:
            m.add(L.Dropout(.2));m.add(L.BatchNormalization())
        if i in (2,5): m.add(L.MaxPooling2D((2,2)))
    m.add(L.Flatten())
    for _ in range(3):
        m.add(L.Dense(750));m.add(L.Activation('relu'));m.add(L.Dropout(.3))
    m.add(L.Activation('relu'));m.add(L.Dense(250));m.add(L.Dropout(.2))
    m.add(L.BatchNormalization());m.add(L.Activation('relu'))
    m.add(L.Dense(64));m.add(L.Activation('relu'))
    m.add(L.Dense(7,activation='softmax'))
    return m


def create_model(name):
    if name=='compact': return compact_model()
    if name=='spatial': return compact_model(spatial=True)
    if name=='original': return original_final_model()
    if name=='baseline':
        from expression import build_model
        return build_model()
    raise ValueError('Unknown architecture: '+name)
