import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt


class AutoEncoderModel:
    def __init__(self, input_dim, encoding_dim=32, l1=1e-5, l2=1e-4, epochs=100, batch_size=256, learning_rate=0.001):
        """
        Initialize the autoencoder with input and encoding dimensions and regularization parameters.
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.l1 = l1
        self.l2 = l2
        self.autoencoder = None
        self.encoder_model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


    def build_autoencoder(self):
        """
        Build the autoencoder model with elastic net regularization.
        """
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(input_layer)
        encoder = Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(encoder)
        encoder = Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(encoder)
        bottleneck = Dense(self.encoding_dim, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(encoder)
        decoder = Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(bottleneck)
        decoder = Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(decoder)
        decoder = Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(decoder)
        output_layer = Dense(self.input_dim, activation="sigmoid", kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2))(decoder)

        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        self.encoder_model = Model(inputs=input_layer, outputs=bottleneck)

    def train_autoencoder(self, X_train):
        """
        Train the autoencoder with the training data.
        """
        history = self.autoencoder.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_split=0.2)
        return history

    def get_encoded_features(self, X):
        """
        Get the encoded features from the trained encoder model.
        """
        return self.encoder_model.predict(X)

    def plot_training_history(self, history, path):
        """
        Plot the training and validation loss/accuracy over epochs.
        """
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if path:
            plt.savefig(path+'training_loss_history.png')

    


if __name__ == '__main__':
    pass