# Cats vs Dogs Classification using-CNN

This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset used is from the [Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats).

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Dataset

The dataset consists of images of cats and dogs. The data is split into training and testing sets:

- `train.zip`: Contains 25,000 labeled images of cats and dogs.
- `test1.zip`: Contains 12,500 unlabeled images for prediction.

## Installation

To run this project, you need to have Python 3.x and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Matplotlib
- scikit-learn

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy pandas pillow matplotlib scikit-learn


Data Preparation
Download the dataset from Kaggle and unzip the files.

python
Copier le code
from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c dogs-vs-cats
Unzip the downloaded files:

python
Copier le code
import zipfile

zip_path = "/content/dogs-vs-cats.zip"
extract_path = "/content/dogs-vs-cats"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
Extract training and testing images:

python
Copier le code
files_to_extract = ['train.zip', 'test1.zip']

for file in files_to_extract:
    file_path = os.path.join('/content/dogs-vs-cats', file)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
Model Architecture
The model architecture consists of multiple convolutional layers followed by batch normalization, max pooling, and dropout layers. Here is a summary:

python
Copier le code
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf

class CatVSDog(tf.keras.Model):
    def __init__(self, filter_size, Image_Width, Image_Height, Image_Channels, l2=0, dropout_val=0.0):
        super(CatVSDog, self).__init__()

        self.conv_layers = []
        self.norm_layers = []
        self.pool_layers = []
        self.dropout_layers = []

        for i in range(3):
            self.conv_layers.append(Conv2D(filter_size * 2**i, (3, 3), activation='relu'))
            self.norm_layers.append(BatchNormalization())
            self.pool_layers.append(MaxPooling2D(pool_size=(2, 2)))
            self.dropout_layers.append(Dropout(dropout_val))

        self.flatten = Flatten()
        self.dense1 = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(l2), activation='relu')
        self.dropout = Dropout(dropout_val*2)
        self.dense2 = Dense(2, activation='softmax', name='output')

    def call(self, inputs):
        x = inputs

        for i in range(3):
            x = self.conv_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)

        return self.dense2(x)
Training
Compile the model with the specified optimizer and loss function:

python
Copier le code
model = CatVSDog(16, Image_Width, Image_Height, Image_Channels, 1e-6, 0.25)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.Precision()])
Prepare data generators for training and validation:

python
Copier le code
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(train_df, "/content/dogs-vs-cats/train/", x_col='filename', y_col='category', target_size=Image_Size, class_mode='categorical', batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, "/content/dogs-vs-cats/train/", x_col='filename', y_col='category', target_size=Image_Size, class_mode='categorical', batch_size=batch_size)
Train the model:

python
Copier le code
epochs = 10
history = model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=total_validate//batch_size, steps_per_epoch=total_train//batch_size, callbacks=callbacks)
Evaluation
After training, you can evaluate the model using the validation dataset. Ensure to save the model for future use:

python
Copier le code
model.save("model_catsVSdogs", save_format='tf')
Results
The results of the training and evaluation phases will be displayed in the terminal. You can plot the accuracy and loss over epochs to visualize the performance.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

css
Copier le code

This README file includes all necessary sections and details about the project, from installation to resul
