from keras.models                import Sequential
from keras.layers                import Activation, Dropout, Flatten, Dense

def ModelSVM(input_shape, optimizer, hidden_layers=[256], dropout=0.5):
    
    # Input Layers
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    
    # Hidden Layers
    for hl in hidden_layers:
        model.add(Dense(hl, activation='relu'))
        model.add(Dropout(dropout))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    
    return model