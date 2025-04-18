from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam

def create_gender_model():
    input_shape = (48, 48, 1)
    inputs = Input(shape=input_shape)
    
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout1 = Dropout(0.25)(pool1)
    
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(dropout1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    dropout2 = Dropout(0.25)(pool3)
    
    flatten = Flatten()(dropout2)
    dense1 = Dense(1024, activation='relu')(flatten)
    dropout3 = Dropout(0.5)(dense1)
    
    # Gender output (binary classification)
    gender_output = Dense(1, activation='sigmoid', name='gender_out')(dropout3)
    
    # Define model
    model = Model(inputs=inputs, outputs=gender_output)
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

def load_gender_model(weights_path):
    model = create_gender_model()
    model.load_weights(weights_path)
    return model