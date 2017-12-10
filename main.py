import cv2
import random
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
from keras.optimizers import Adam
from claptcha import Claptcha


all_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
captcha_length = 6
batch_size = 64


class CaptchaAccuracyCallback(Callback):
    def __init__(self, model, test_batch_count=100):
        self.model = model
        self.test_batch_count = test_batch_count

        c = Claptcha('g0vt1k', 'arial.ttf')
        text, image = c.image

        self.img_array = np.asarray(image)

    def on_epoch_end(self, epoch, logs=None):
        correct_count = 0
        total_count = self.test_batch_count * batch_size

        correct_character_count = 0
        total_character_count = self.test_batch_count * batch_size * captcha_length

        if epoch == 0 or epoch == 9 or epoch == 19 or epoch == 29:
            prediction = model.predict(self.img_array.reshape(1, 80, 200, 3))[0]
            predicted_text = decode_text(prediction)

            filename = 'predicted_%d_%s.png' % (epoch, predicted_text)
            cv2.imwrite(filename, self.img_array)

        for i in range(0, self.test_batch_count):
            x, y_label = next(data_generator())
            y_pred = model.predict(x)


            for j in range(0, batch_size):
                ground_truth = decode_text(y_label[j])
                guess = decode_text(y_pred[j])

                if ground_truth == guess:
                    correct_count += 1

                for char_index, c in enumerate(ground_truth):
                    if ground_truth[char_index] == guess[char_index]:
                        correct_character_count += 1

        accuracy = (correct_count / total_count) * 100
        accuracy_characters = (correct_character_count / total_character_count) * 100

        print('\n\nAccuracy = %0.2f%%\n' % accuracy)
        print('\n\nCharacter Accuracy = %0.2f%%\n' % accuracy_characters)


def random_string():
    to_return = ''

    for i in range(0, captcha_length):
        rand_index = random.randint(0, len(all_chars) - 1)
        to_return += all_chars[rand_index]

    return to_return


def random_font():
    fonts = ['arial.ttf'] #, 'gigi.ttf', 'kunstler.ttf']
    rand_index = random.randint(0, len(fonts) - 1)
    return fonts[rand_index]


def encode_text(text):
    encoding = np.zeros(shape=(len(text), len(all_chars)))
    for str_index, char in enumerate(text):
        char_index = all_chars.index(char)
        encoding[str_index, char_index] = 1

    return encoding.flatten()


def decode_text(encoding):
    text = ''
    encoded_mat = encoding.reshape(captcha_length, -1)

    for i in range(0, captcha_length):
        vector = encoded_mat[i]
        char_index = np.argmax(vector)
        text += all_chars[char_index]

    return text


def data_generator():
    while True:
        x_batch = []
        y_batch = []
        for i in range(0, batch_size):
            rand_string = random_string()
            rand_font = random_font()

            c = Claptcha(rand_string, rand_font)
            text, image = c.image

            x = np.asarray(image)
            y = encode_text(rand_string)

            x_batch.append(x)
            y_batch.append(y)

        yield np.array(x_batch), np.array(y_batch)


def loss_for_captcha(y_true, y_pred):
    y_true_mat = K.reshape(y_true, (captcha_length, -1))
    y_pred_mat = K.reshape(y_pred, (captcha_length, -1))

    loss = 0
    for i in range(0, captcha_length):
        y_letter_true = y_true_mat[i, :]
        y_letter_pred = y_pred_mat[i, :]

        loss += categorical_crossentropy(y_letter_true, y_letter_pred)

    return loss


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 200, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(len(all_chars) * captcha_length, activation='softmax'))

model.compile(loss=loss_for_captcha, optimizer=Adam(decay=1e-5))

accuracy_callback = CaptchaAccuracyCallback(model)
model.fit_generator(data_generator(), steps_per_epoch=100, epochs=30, verbose=1, callbacks=[accuracy_callback])

x_batch, y_batch = next(data_generator())
y_predicted = model.predict(x_batch)
for i in range(0, 10):
    image = x_batch[i]
    text = decode_text(y_predicted[i])

    cv2.imwrite('solved/[' + str(i) + '] ' + text + '.png', image)
