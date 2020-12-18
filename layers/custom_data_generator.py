import numpy as np
import tensorflow as tf
import os


class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, directory, batch_size=5, dim=(224, 224), channels=3, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.labels = {}
        self.files = []
        self.shuffle = shuffle
        self.on_epoch_end()
        for file in os.listdir(directory + "/NORMAL"):
            self.labels[file] = 0
            self.files.append(file)
        for file in os.listdir(directory + "/PNEUMONIA"):
            self.labels[file] = 1
            self.files.append(file)

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        X_test, y_test = self.__convert(batch)
        return X_test, y_test

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)

    def __convert(self, batch):
        X_test = np.empty((self.batch_size, *self.dim, self.channels))
        y_test = np.empty(self.batch_size, dtype=int)

        for i, file in enumerate(batch):
            if self.labels[file] == 0:
                single_ch_img = np.genfromtxt(self.directory + "/NORMAL/" + file)
                X_test[i, ] = np.expand_dims(np.stack((single_ch_img,) * self.channels, axis=-1), axis=0)
                y_test[i] = self.labels[file]
            else:
                single_ch_img = np.genfromtxt(self.directory + "/PNEUMONIA/" + file)
                X_test[i, ] = np.expand_dims(np.stack((single_ch_img,) * self.channels, axis=-1), axis=0)
                y_test[i] = self.labels[file]
            return X_test, y_test


if __name__ == '__main__':
    # test
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential

    directory = "/Users/camilla/Desktop/project_2020/encoded"

    params = {'dim': (224, 224),
              'batch_size': 1,
              'channels': 1,
              'shuffle': True}

    training_generator = CustomDataGenerator(directory, **params)

    model = Sequential([
        tf.keras.layers.Dense(2, activation="relu", name="layer1"),
        tf.keras.layers.Dense(3, activation="relu", name="layer2"),
        tf.keras.layers.Dense(4, name="layer3"),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(x=training_generator, )
