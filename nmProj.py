import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
tf.get_logger().setLevel(logging.ERROR)
from keras.utils import image_dataset_from_directory
from keras import layers
from keras import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


main_path='./data/'

img_size=(64,64)
batch_size=64

#5. zadatak - podela na trening, validaciju

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval_test = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

val_test_batch= len(Xval_test)
val_batch =  int(0.5 * val_test_batch)

Xval = Xval_test.take(val_batch)
Xtest = Xval_test.skip(val_batch)


classes = Xtrain.class_names
print(classes)

#4.zadatak - prikaz po jednog podatka iz svake klase

N = 4
j = 0
i = 0
plt.figure()
for img, lab in Xtrain:
    if classes[lab[i]] == classes[j]:
        plt.subplot(1, N, j+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
        j = j+1
    i = i+1
    if(j==4): break 
plt.show()

#3.zadatak - histogram, balansiranost

k1 = os.listdir('./data/cloudy/')
k2 = os.listdir('./data/desert/')
k3 = os.listdir('./data/green_area/')
k4 = os.listdir('./data/water/')
for i in range(len(k1)):
 k1[i] = 0
for i in range(len(k2)):
 k2[i] = 1
for i in range(len(k3)):
 k3[i] = 2
for i in range(len(k4)):
 k4[i] = 3
br1 = pd.DataFrame(k1)
br2 = pd.DataFrame(k2)
br3 = pd.DataFrame(k3)
br4 = pd.DataFrame(k4)
Y = pd.concat((br1, br2, br3, br4), axis=0)
#plt.figure()
Y.hist()

plt.xticks(np.arange(0, N ,1), classes)
plt.xlabel('Klase')
plt.ylabel('Broj odbiraka')
plt.title('Histogram klasa')
plt.grid(False)
plt.show()
   

#6.zadatak - pretprocesiranje podataka

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
    ]
)

#7.zadatak - formiranje i obucavanje neuralne mreze

num_classes = len(classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.summary()
model.compile(Adam(learning_rate=0.001),
                loss=SparseCategoricalCrossentropy(),
                metrics='accuracy')

#8. zadatak - zastita od preobucavanja (rano zaustavljanje)

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

#10. zadatak - grafik performanse
history = model.fit(Xtrain,
                    epochs=1000,
                    validation_data=Xval,
                    callbacks=[es],
                    verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

#10. zadatak - matrica konfuzije na trening skupu

trueTrain = np.array([])
predTrain = np.array([])

for img, label in Xtrain:
    trueTrain = np.append(trueTrain, label)
    predTrain = np.append(predTrain, np.argmax(model.predict(img, verbose=0), axis=1))

cm = confusion_matrix(trueTrain, predTrain, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

print('Tačnost modela na trening skupu je: ' + str(100 * accuracy_score(trueTrain, predTrain)))

#10. zadatak - matrica konfuzije na test skupu

trueTest = np.array([])
predTest = np.array([])

for img, label in Xtest:
    trueTest = np.append(trueTest, label)
    predTest = np.append(predTest, np.argmax(model.predict(img, verbose=0), axis=1))

cm = confusion_matrix(trueTest, predTest, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

print('Tačnost modela na test skupu je: ' + str(100 * accuracy_score(trueTest, predTest)))


#10. zadatak - primeri dobro i lose klasifikovanih primera

N = 10
cnt = 0
labels = np.array([])
for img, label in Xtrain:
    labels = np.append(labels, label)
    pred = np.argmax(model.predict(img, verbose=0), axis=1)
    for i in range(len(label)):
        if classes[label[i]] == classes[pred[i]]:
            cnt += 1
            if cnt > N:
                break
            plt.subplot(2, int(N/2), cnt)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(classes[pred[i]])
            plt.axis('off')
plt.show()

cnt = 0
labels = np.array([])
for img, label in Xtrain:
    labels = np.append(labels, label)
    pred = np.argmax(model.predict(img, verbose=0), axis=1)
    for i in range(len(label)):
        if classes[label[i]] != classes[pred[i]]:
            cnt += 1
            if cnt > N:
                break
            plt.subplot(2, int(N/2), cnt)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(classes[pred[i]])
            plt.axis('off')
plt.show()