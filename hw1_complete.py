#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

#print(f"TensorFlow Version: {tf.__version__}")
#print(f"Keras Version: {keras.__version__}")

def build_model1():
    model = Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(10)
    ])

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

def build_model2():
    model = Sequential([
        layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)
    ])

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

def build_model3():

    model = Sequential([

        layers.SeparableConv2D(32, 3, padding="same", activation="relu",
                                input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)
    ])

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model

def build_model50k():
    return keras.models.load_model("best_model.h5", compile=True)

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import

if __name__ == '__main__':
  RUN_TRAINING_1 = False
  RUN_TRAINING_2 = True
  RUN_TRAINING_3 = False
  RUN_TRAINING_50K = False

  ###################################################################
  ## Add code here to Load the CIFAR10 data set

  (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

  train_images = train_images.astype("float32") / 255.0
  test_images  = test_images.astype("float32") / 255.0

  train_labels = train_labels.squeeze()
  test_labels  = test_labels.squeeze()

  val_images  = train_images[-5000:]
  val_labels  = train_labels[-5000:]

  train_images = train_images[:-5000]
  train_labels = train_labels[:-5000]

  #MODEL1##################################################################

  model1 = build_model1()
  model1.summary()

  if RUN_TRAINING_1:
    history1 = model1.fit(
      train_images, train_labels,
      epochs = 30,
      validation_data = (val_images, val_labels)
    )

    train_loss, train_acc = model1.evaluate(train_images, train_labels, verbose=0)
    val_loss, val_acc     = model1.evaluate(val_images, val_labels, verbose=0)
    test_loss, test_acc   = model1.evaluate(test_images, test_labels, verbose=0)

    print("Model1 train acc:", train_acc)
    print("Model1 val acc:  ", val_acc)
    print("Model1 test acc: ", test_acc)

    #MODEL2##################################################################

    model2 = build_model2()
    model2.summary()

    if RUN_TRAINING_2:
        history2 = model2.fit(
            train_images, train_labels,
            epochs=30,
            validation_data=(val_images, val_labels)
        )

    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    img = np.array(keras.utils.load_img(
        "test_image_cat.jpg",
        color_mode="rgb",
        target_size=(32, 32)
    )) / 255.0

    pred = model2.predict(img[None, ...])
    pred_idx = np.argmax(pred)

    print("Predicted class:", class_names[pred_idx])

    train_loss2, train_acc2 = model2.evaluate(train_images, train_labels, verbose=0)
    val_loss2, val_acc2     = model2.evaluate(val_images, val_labels, verbose=0)
    test_loss2, test_acc2   = model2.evaluate(test_images, test_labels, verbose=0)

    print("Model2 train acc:", train_acc2)
    print("Model2 val acc:  ", val_acc2)
    print("Model2 test acc: ", test_acc2)

    #MODEL3##################################################################

    model3 = build_model3()
    model3.summary()

    if RUN_TRAINING_3:
        history3 = model3.fit(
            train_images, train_labels,
            epochs=30,
            validation_data=(val_images, val_labels)
        )

    train_loss3, train_acc3 = model3.evaluate(train_images, train_labels, verbose=0)
    val_loss3, val_acc3     = model3.evaluate(val_images, val_labels, verbose=0)
    test_loss3, test_acc3   = model3.evaluate(test_images, test_labels, verbose=0)

    print("Model3 train acc:", train_acc3)
    print("Model3 val acc:  ", val_acc3)
    print("Model3 test acc: ", test_acc3)

    ###################################################################

    model50k = build_model50k()
    model50k.summary()

    if RUN_TRAINING_50K:
        model50k.fit(
            train_images, train_labels,
            epochs=30,
            validation_data=(val_images, val_labels)
        )

    model50k.save("best_model.h5")

    train_loss50, train_acc50 = model50k.evaluate(train_images, train_labels, verbose=0)
    val_loss50,   val_acc50   = model50k.evaluate(val_images, val_labels, verbose=0)
    test_loss50,  test_acc50  = model50k.evaluate(test_images, test_labels, verbose=0)

    print("50k model train acc:", train_acc50)
    print("50k model val acc:  ", val_acc50)
    print("50k model test acc: ", test_acc50)