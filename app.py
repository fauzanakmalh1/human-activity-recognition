import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import itertools
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from PIL import Image, ImageOps
from moviepy.editor import *
from collections import deque

FPS = 25
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
CLASSES_LIST = ['No Cheat', 'Read Text', 'Ask Friend', 'Call Friend']
MODEL_OUTPUT_SIZE = len(CLASSES_LIST)
BASE_DIR_ASSET = './asset'
BASE_DIR_BEST = './model_best'
BASE_DIR_NEW = './model_new'
MODEL_PATH_BEST = f'{BASE_DIR_BEST}/checkpoint/HAR_MobileNetV2_Model_Best.h5'
OUTPUT_DIRECTORY = f'{BASE_DIR_NEW}/media'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_fcn=st.write):
        tf.keras.callbacks.Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_begin(self, epoch, logs={}):
        msg = f'Menjalankan Epoch {epoch + 1}'
        self.print_fcn(msg)

    def on_epoch_end(self, epoch, logs={}):
        values = ' - '.join(
            '{}: {:0.4f}'.format(k, logs[k]) for k in logs)
        msg = f'Epoch {epoch + 1}: {values}'
        self.print_fcn(msg)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def get_model_best():
    model = load_model(MODEL_PATH_BEST, compile=False)
    return model


def image_classification(filename):
    model = get_model_best()
    size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    img = ImageOps.fit(filename, size, Image.ANTIALIAS)
    img = img_to_array(img)
    img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    img = img.astype('float32')
    img = img / 255.0
    prediction = model.predict(img)
    prediction = prediction[0]
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(prediction)[
        ::-1]

    st.write('##### Prediksi Kelas dan Probabilitas-nya pada Foto:')
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
        predicted_class_name = CLASSES_LIST[predicted_label]
        predicted_probability = prediction[predicted_label] * 100
        if predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[0]:
            st.write(
                f'##### {predicted_class_name}: {predicted_probability:.2f}%')
        else:
            st.write(
                f'{predicted_class_name}: {predicted_probability:.2f}%')
    return True


def video_classification(video_file_path, output_file_path):
    model = get_model_best()
    predicted_labels_probabilities_deque = deque(maxlen=FPS)
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), FPS, (original_video_width, original_video_height))

    while True:
        status, frame = video_reader.read()

        if not status:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        predicted_labels_probabilities = model.predict(
            np.expand_dims(normalized_frame, axis=0))[0]
        predicted_labels_probabilities_deque.append(
            predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == FPS:
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque)
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(
                axis=0)
            predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(
                predicted_labels_probabilities_averaged)[::-1]

            for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
                predicted_class_name = CLASSES_LIST[predicted_label]
                predicted_probability = predicted_labels_probabilities_averaged[
                    predicted_label] * 100
                if predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[0]:
                    cv2.putText(frame, 'Prediksi Kelas:', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 69, 255), 2)
                    cv2.putText(frame, f'{predicted_class_name} ({predicted_probability:.2f}%)',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                elif predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[1]:
                    cv2.putText(frame, f'{predicted_class_name} ({predicted_probability:.2f}%)',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 69, 255), 2)
                elif predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[2]:
                    cv2.putText(frame, f'{predicted_class_name} ({predicted_probability:.2f}%)',
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 69, 255), 2)
                elif predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[3]:
                    cv2.putText(frame, f'{predicted_class_name} ({predicted_probability:.2f}%)',
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 69, 255), 2)

        video_writer.write(frame)

    # video_reader.release()
    video_writer.release()

    predicted_labels_probabilities_np = np.zeros(
        (FPS, MODEL_OUTPUT_SIZE), dtype=np.float64)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = video_frames_count // FPS

    for frame_counter in range(FPS):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,
                         frame_counter * skip_frames_window)
        _, frame = video_reader.read()
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        predicted_labels_probabilities = model.predict(
            np.expand_dims(normalized_frame, axis=0))[0]
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(
        axis=0)
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(
        predicted_labels_probabilities_averaged)[::-1]

    st.write('##### Summary Prediksi Kelas dan Probabilitas-nya pada Video:')
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
        predicted_class_name = CLASSES_LIST[predicted_label]
        predicted_probability = predicted_labels_probabilities_averaged[predicted_label] * 100
        if predicted_label == predicted_labels_probabilities_averaged_sorted_indexes[0]:
            st.write(
                f'##### {predicted_class_name}: {predicted_probability:.2f}%')
        else:
            st.write(
                f'{predicted_class_name}: {predicted_probability:.2f}%')

    video_reader.release()
    return True


def get_model_name(fold_var_new):
    return f'{BASE_DIR_NEW}/checkpoint/HAR_MobileNetV2_Model_fold-{str(fold_var_new)}.h5'


def get_model(fold_var_new, dense_layer_new, init_lr_new, epochs_new):
    st.write('---')
    st.write(f'##### [INFO] Membangun Model Fold-{str(fold_var_new)}')
    baseModel = MobileNetV2(weights='imagenet',
                            include_top=False,
                            input_tensor=Input(
                                shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
                            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                            classes=MODEL_OUTPUT_SIZE)
    baseModel.trainable = False

    headModel = baseModel.output
    headModel = Conv2D(100, (3, 3), activation='relu', input_shape=(
        IMAGE_HEIGHT, IMAGE_WIDTH, 3))(headModel)
    headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name='flatten')(headModel)
    if dense_layer_new == 1:
        headModel = Dense(512, activation='relu',
                          name='dense_layer_1')(headModel)
    elif dense_layer_new == 3:
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_3')(headModel)
    elif dense_layer_new == 5:
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_3')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_4')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_5')(headModel)
    elif dense_layer_new == 7:
        headModel = Dense(4096, activation='relu',
                          name='dense_layer_1')(headModel)
        headModel = Dense(4096, activation='relu',
                          name='dense_layer_2')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_3')(headModel)
        headModel = Dense(2048, activation='relu',
                          name='dense_layer_4')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_5')(headModel)
        headModel = Dense(1024, activation='relu',
                          name='dense_layer_6')(headModel)
        headModel = Dense(512, activation='relu',
                          name='dense_layer_7')(headModel)
    headModel = Dense(MODEL_OUTPUT_SIZE, activation='softmax',
                      name='dense_layer_out')(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    opt = Adam(learning_rate=init_lr_new, decay=init_lr_new / epochs_new)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def plot_history(H, fold_var_new):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, len(H.history['loss'])+1),
             H.history['loss'], label='train_loss')
    plt.plot(range(1, len(H.history['val_loss'])+1),
             H.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    savefig_dir = f'{BASE_DIR_NEW}/plot/plot_loss_fold-{str(fold_var_new)}.png'
    plt.savefig(savefig_dir, bbox_inches='tight')

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, len(H.history['accuracy'])+1),
             H.history['accuracy'], label='train_acc')
    plt.plot(range(1, len(H.history['val_accuracy'])+1),
             H.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    savefig_dir = f'{BASE_DIR_NEW}/plot/plot_accuracy_fold-{str(fold_var_new)}.png'
    plt.savefig(savefig_dir, bbox_inches='tight')

    report_df = pd.DataFrame({
        'loss': H.history['loss'],
        'val_loss': H.history['val_loss'],
        'accuracy': H.history['accuracy'],
        'val_accuracy': H.history['val_accuracy']
    })
    report_df.to_csv(
        f'{BASE_DIR_NEW}/report/HAR_accuracy_loss_report_fold-{str(fold_var_new)}.csv')


def plot_confusion_matrix(cm, classes, fold_var_new, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    thresh = cm.max()/2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if normalize:
        savefig_dir = f'{BASE_DIR_NEW}/plot/plot_confusion_matrix_normalized_fold-{str(fold_var_new)}.png'
        plt.savefig(savefig_dir, bbox_inches='tight')
    else:
        st.write('---')
        st.write(f'##### [INFO] Confusion Matrix Fold-{str(fold_var_new)}')
        savefig_dir = f'{BASE_DIR_NEW}/plot/plot_confusion_matrix_fold-{str(fold_var_new)}.png'
        plt.savefig(savefig_dir, bbox_inches='tight')
        image = Image.open(savefig_dir)
        st.image(image, use_column_width=True)


st.set_page_config(page_title='Human Activity Recognition',
                   page_icon=':computer:')
st.set_option('deprecation.showfileUploaderEncoding', False)

local_css('./style/style.css')
model = get_model_best()

st.sidebar.subheader(
    '*Human Activity Recognition* Berdasarkan Tangkapan *Webcam* Menggunakan Metode MobileNet')

st.sidebar.write('---')
menu_name = st.sidebar.selectbox(
    'Silahkan Pilih Menu', ['Halaman Depan', 'Simulasi Pelatihan Model', 'Hasil Model Terbaik', 'Prediksi Foto', 'Prediksi Video'])

if menu_name == 'Halaman Depan':
    st.write('## :house: Halaman Depan')

    st.write('---')
    st.write('##### Judul Skripsi:')
    st.write(
        '#### *Human Activity Recognition* Berdasarkan Tangkapan *Webcam* Menggunakan Metode MobileNet')

    st.write('---')
    st.write('##### Abstrak:')
    st.markdown('<div style="text-align: justify;"><p>&nbsp;&nbsp;&nbsp;&nbsp;Manusia tidak bisa terlepas dari aktivitas sehari-hari yang mana merupakan bagian dari aktivitas kehidupan manusia. <i>Human Activity Recognition</i> (HAR) atau pengenalan aktivitas manusia saat ini merupakan salah satu topik yang sedang banyak diteliti seiring dengan pesatnya kemajuan di bidang teknologi yang berkembang saat ini. Hampir semua bidang terdampak dari pandemi COVID-19 yang mempengaruhi aktivitas manusia sehingga menjadi lebih terbatas. Salah satu bidang yang paling terdampak yaitu pendidikan, di mana kampus menerapkan sistem pembelajaran daring, yang membuat dosen lebih sulit untuk mengawasi pembelajaran maupun ujian yang dilakukan secara daring karena tidak dapat mengawasi aktivitas yang dilakukan mahasiswa secara langsung.</p><p>&nbsp;&nbsp;&nbsp;&nbsp;Penelitian ini bertujuan untuk mengetahui prediksi aktivitas yang dilakukan oleh seseorang saat ujian daring berdasarkan tangkapan <i>webcam</i> dengan memanfaatkan model <i>deep learning</i> dengan metode <i>Convolution Neural Network</i> menggunakan arsitektur MobileNetV2. Penelitian dilakukan menggunakan dataset berupa aktivitas yang terbagi menjadi empat kelas yang dilakukan oleh seseorang saat melaksanakan ujian daring, yang diolah melalui tahap <i>preprocessing</i>, pelatihan model, evaluasi model, dan implementasi aplikasi.</p><p>&nbsp;&nbsp;&nbsp;&nbsp;Dari penelitian tersebut, dapat diprediksi aktivitas yang dilakukan oleh seseorang saat melaksanakan ujian daring dengan hasil terbaik diraih oleh model dengan <i>hyperparameter</i> yang diuji berupa jumlah <i>dense layer</i> sebanyak 5 dan jumlah <i>batch size</i> sebesar 16. Model tersebut berhasil memberikan performa <i>F1-score</i> akhir sebesar 84,52%.</p></div>', unsafe_allow_html=True)

    st.write('---')
    st.write('##### Dataset:')
    st.markdown('<div style="text-align: justify;"><p>&nbsp;&nbsp;&nbsp;&nbsp;Data yang digunakan dalam penelitian ini berupa data video yang merupakan rekaman dari webcam sejumlah orang yang melaksanakan aktivitas ujian daring. Aktivitas ujian daring yang ada di dalam video yang digunakan pada penelitian ini yaitu aktivitas tidak mencontek atau tidak curang <b>(No Cheat)</b>, membaca teks <b>(Read Text)</b>, bertanya ke teman <b>(Ask Friend)</b>, dan menelpon teman <b>(Call Friend)</b> yang dapat dilihat pada contoh foto berikut.</p></div>', unsafe_allow_html=True)

    with st.expander('Contoh Dataset'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write('No Cheat')
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class0_1.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class0_2.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
        with col2:
            st.write('Read Text')
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class1_1.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class1_2.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
        with col3:
            st.write('Ask Friend')
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class2_1.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class2_2.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
        with col4:
            st.write('Call Friend')
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class5_1.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            image_path = f'{BASE_DIR_ASSET}/image/dataset_class5_2.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)

    st.write('---')
    st.write('##### Tentang Pengembang:')
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander('Penulis'):
            image_path = f'{BASE_DIR_ASSET}/image/penulis_crop.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            st.write('Nama: Fauzan Akmal Hariz')
            st.write('NPM: 140810180005')
    with col2:
        with st.expander('Pembimbing 1'):
            image_path = f'{BASE_DIR_ASSET}/image/pembimbing1_crop.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            st.write('Nama: Dr. Intan Nurma Yulita, MT.')
            st.write('NIP: 19850704 201504 2 003')
    with col3:
        with st.expander('Pembimbing 2'):
            image_path = f'{BASE_DIR_ASSET}/image/pembimbing2_crop.jpg'
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            st.write('Nama: Ino Suryana, Drs., M.Kom.')
            st.write('NIP: 19600115 198701 1 002')

elif menu_name == 'Simulasi Pelatihan Model':
    st.write('## :cd: Simulasi Pelatihan dan Pengujian Model')

    st.write('---')
    st.warning('Catatan: Data yang digunakan pada simulasi pelatihan dan pengujian model pada halaman ini menggunakan 10% dari data yang digunakan pada penelitian.')

    st.write('---')
    st.write('##### Atur Hyperparameter')
    init_lr_new = st.select_slider(
        'Learning Rate', options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], value=1e-1)
    epochs_new = st.slider('Epochs', min_value=2,
                           max_value=40, value=2, step=1)
    early_stopping_new = st.slider('Early Stopping Patience', min_value=1,
                                   max_value=40, value=10, step=1)
    n_split_fold_new = st.slider(
        'K-Fold Split', min_value=2, max_value=20, value=2, step=1)
    batch_size_new = st.select_slider(
        'Batch Size', options=[8, 16, 32, 64, 128, 256], value=8)
    dense_layer_new = st.select_slider(
        'Dense Layer', options=[1, 3, 5, 7], value=1)

    st.write('---')
    st.write(f'''\n##### Cek Hyperparameter yang Diatur
        \nLearning Rate: {init_lr_new}
        \nEpochs: {epochs_new}
        \nEarly Stopping: {early_stopping_new}
        \nK-Fold Split: {n_split_fold_new}
        \nBatch Size: {batch_size_new}
        \nDense Layer: {dense_layer_new}''', unsafe_allow_html=True)
    if dense_layer_new == 1:
        st.code(
            f'''Dense Layer yang Digunakan:\nDense(512, activation='relu', name='dense_layer_1')''', language='python')
    elif dense_layer_new == 3:
        st.code(f'''Dense Layer yang Digunakan:\nDense(1024, activation='relu', name='dense_layer_1')\nDense(1024, activation='relu', name='dense_layer_2')\nDense(512, activation='relu', name='dense_layer_3')''', language='python')
    elif dense_layer_new == 5:
        st.code(f'''Dense Layer yang Digunakan:\nDense(2048, activation='relu', name='dense_layer_1')\nDense(2048, activation='relu', name='dense_layer_2')\nDense(1024, activation='relu', name='dense_layer_3')\nDense(1024, activation='relu', name='dense_layer_4')\nDense(512, activation='relu', name='dense_layer_5')''', language='python')
    elif dense_layer_new == 7:
        st.code(f'''Dense Layer yang Digunakan:\nDense(4096, activation='relu', name='dense_layer_1')\nDense(4096, activation='relu', name='dense_layer_2')\nDense(2048, activation='relu', name='dense_layer_3')\nDense(2048, activation='relu', name='dense_layer_4')\nDense(1024, activation='relu', name='dense_layer_5')\nDense(1024, activation='relu', name='dense_layer_6')\nDense(512, activation='relu', name='dense_layer_7')''', language='python')

    st.write('---')
    if st.button('Jalankan Model'):
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=[0.9, 1.0],
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = pd.read_csv(
            f'{BASE_DIR_NEW}/data/split_perdata/train_labels.csv')
        test_data = pd.read_csv(
            f'{BASE_DIR_NEW}/data/split_perdata/test_labels.csv')

        train_y = train_data.label
        train_x = train_data.drop(['label'], axis=1)

        skf = StratifiedKFold(n_splits=n_split_fold_new,
                              shuffle=True, random_state=47)
        data_kfold = pd.DataFrame()

        validation_accuracy = []
        validation_loss = []
        fold_var_new = 1

        for train_index, val_index in list(skf.split(train_x, train_y)):
            training_data = train_data.iloc[train_index]
            validation_data = train_data.iloc[val_index]

            train_data_generator = train_datagen.flow_from_dataframe(
                training_data,
                directory=f'{BASE_DIR_NEW}/data/split_perdata/train/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=True)

            valid_data_generator = train_datagen.flow_from_dataframe(
                validation_data,
                directory=f'{BASE_DIR_NEW}/data/split_perdata/train/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=True)

            model = get_model(fold_var_new, dense_layer_new,
                              init_lr_new, epochs_new)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                get_model_name(fold_var_new),
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max')
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_new,
                verbose=1,
                mode='auto',
                baseline=None)
            log_callback = LoggingCallback()

            placeholder = st.empty()
            placeholder.write('Pelatihan dan Pengujian Model Sedang Diproses')

            with st.spinner('Silahkan Tunggu ...'):
                history = model.fit(train_data_generator,
                                    steps_per_epoch=train_data_generator.samples // train_data_generator.batch_size,
                                    epochs=epochs_new,
                                    validation_data=valid_data_generator,
                                    validation_steps=valid_data_generator.samples // valid_data_generator.batch_size,
                                    verbose=1,
                                    callbacks=[checkpoint, early_stopping, log_callback])
                plot_history(history, fold_var_new)

            placeholder.empty()

            model.load_weights(
                f'{BASE_DIR_NEW}/checkpoint/HAR_MobileNetV2_Model_fold-{str(fold_var_new)}.h5')

            test_data_generator = test_datagen.flow_from_dataframe(
                test_data,
                directory=f'{BASE_DIR_NEW}/data/split_perdata/test/',
                x_col='filename',
                y_col='label',
                target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                color_mode='rgb',
                class_mode='categorical',
                batch_size=batch_size_new,
                shuffle=False)

            test_data_generator.reset()
            test_steps_per_epoch = np.math.ceil(
                test_data_generator.samples / test_data_generator.batch_size)
            predictions = model.predict_generator(
                test_data_generator, steps=test_steps_per_epoch)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())

            cm = confusion_matrix(true_classes, predicted_classes)
            plot_confusion_matrix(
                cm, class_labels, fold_var_new, normalize=True)
            plot_confusion_matrix(
                cm, class_labels, fold_var_new, normalize=False)

            st.write('---')
            st.write(
                f'##### [INFO] Classification Report Fold-{str(fold_var_new)}')
            report = classification_report(
                true_classes, predicted_classes, target_names=class_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(
                f'{BASE_DIR_NEW}/report/HAR_classification_report_fold-{str(fold_var_new)}.csv')
            st.dataframe(report_df)

            data_kfold[fold_var_new] = predicted_classes

            st.write('---')
            st.write(f'##### [INFO] Fold-{fold_var_new} Selesai Dijalankan')
            if fold_var_new == n_split_fold_new:
                data_kfold.to_csv(
                    f'{BASE_DIR_NEW}/report/HAR_data_kfold_new.csv')
                st.write('---')
                st.success('Pelatihan dan Pengujian Model Selesai Dijalankan')
            tf.keras.backend.clear_session()
            fold_var_new += 1

elif menu_name == 'Hasil Model Terbaik':
    st.write('## :floppy_disk: Hasil Pelatihan dan Pengujian Model Terbaik')

    st.write('---')
    init_lr_new = 1e-4
    epochs_new = 20
    early_stopping_new = 10
    n_split_fold_new = 5
    batch_size_new = 16
    dense_layer_new = 5
    st.write(f'''\n##### Hyperparameter
        \nLearning Rate: {init_lr_new}
        \nMax Epochs: {epochs_new}
        \nEarly Stopping: {early_stopping_new}
        \nK-Fold Cross Validation: {n_split_fold_new}
        \nBatch Size: {batch_size_new}
        \nDense Layer: {dense_layer_new}''', unsafe_allow_html=True)
    st.code(f'''Dense Layer yang Digunakan:\nDense(2048, activation='relu', name='dense_layer_1')\nDense(2048, activation='relu', name='dense_layer_2')\nDense(1024, activation='relu', name='dense_layer_3')\nDense(1024, activation='relu', name='dense_layer_4')\nDense(512, activation='relu', name='dense_layer_5')''', language='python')

    st.write('---')
    st.write('##### Confusion Matrix')
    image_path = f'{BASE_DIR_BEST}/plot/plot_confusion_matrix_best.png'
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

    st.write('---')
    st.write('##### Classification Report')
    classication_report_df = pd.read_csv(
        f'{BASE_DIR_BEST}/report/HAR_classification_report_best.csv')
    st.dataframe(classication_report_df)

elif menu_name == 'Prediksi Foto':
    st.write('## :camera: Prediksi Foto')

    st.write('---')
    uploaded_image = st.file_uploader(
        'Silahkan Pilih Foto yang Ingin Diprediksi', type=['jpg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image.save(f'{OUTPUT_DIRECTORY}/image/{uploaded_image.name}')

        st.write('---')
        st.write('##### Foto yang Diupload:')
        st.image(image, use_column_width=True)

        st.write('---')
        placeholder = st.empty()
        placeholder.write('Prediksi Sedang Diproses')

        with st.spinner('Silahkan Tunggu ...'):
            image_classification_run = image_classification(image)

        if image_classification_run:
            placeholder.empty()

elif menu_name == 'Prediksi Video':
    st.write('## :video_camera: Prediksi Video')

    st.write('---')
    uploaded_video = st.file_uploader(
        'Silahkan Pilih Video yang Ingin Diprediksi', type=['mp4', 'mov'])

    if uploaded_video is not None:
        video = uploaded_video.name
        video_name = video.rsplit('.', 1)[0]

        with open(video, mode='wb') as f:
            f.write(uploaded_video.read())
        input_video_file_path = f'{OUTPUT_DIRECTORY}/video/{video}'
        shutil.move(video, input_video_file_path)

        st.write('---')
        st.write('##### Video yang Diupload:')
        st.video(input_video_file_path, start_time=0)
        output_video_file_path = f'{OUTPUT_DIRECTORY}/video/{video_name}-Output-WSize{FPS}.mp4'

        st.write('---')
        placeholder = st.empty()
        placeholder.write('Prediksi Sedang Diproses')

        with st.spinner('Silahkan Tunggu ...'):
            video_classification_run = video_classification(
                input_video_file_path, output_video_file_path)

        if video_classification_run:
            placeholder.empty()

            st.write('---')
            placeholder_real_time = st.empty()
            VideoFileClip(output_video_file_path).ipython_display(
                width=700, autoplay=1, loop=1)
            st.video('./__temp__.mp4', start_time=0)
            placeholder_real_time.write(
                '##### Prediksi Kelas Secara Real Time pada Video:')

            with open(f'{OUTPUT_DIRECTORY}/video/{video_name}-Output-WSize{FPS}.mp4', 'rb') as file:
                btn = st.download_button(
                    label='Download Video',
                    data=file,
                    file_name=f'{video_name}-Output-WSize{FPS}.mp4',
                    mime='video/mp4'
                )

else:
    st.write(
        '## *Human Activity Recognition* Berdasarkan Tangkapan *Webcam* Menggunakan Metode MobileNet')

    st.write('---')
    st.write('##### Silahkan Pilih Menu di *Sidebar*')
