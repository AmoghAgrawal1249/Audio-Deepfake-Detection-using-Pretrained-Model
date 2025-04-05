import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def build_model(input_shape=(128, 128, 3)):
    """ Build a deepfake detection model using EfficientNetB0 for transfer learning. """
    
    # Load pre-trained EfficientNetB0 model without the top layer
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, 
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model layers to retain learned features
    base_model.trainable = False  
    
    # Define the new model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(1, activation='sigmoid')  # Binary classification (bonafide/spoof)
    ])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
if __name__ == "__main__":
    X = np.load("X.npy")
    y = np.load("y.npy")

    # Convert grayscale (128, 128, 1) to 3-channel (128, 128, 3)
    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(input_shape=X.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    model.save("deepfake_detector.h5")
    print("Model saved as deepfake_detector.h5")
