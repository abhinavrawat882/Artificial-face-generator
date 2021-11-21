import os 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

##Fetch Data
pathname="../input/vggface-using-tripletloss/celebs/celebs"
#pathname="../input/vggface-using-tripletloss/crop_faces/faces"
dirList=os.listdir(pathname)
print(dirList)
ln=len(dirList)
print(ln)
pathname+="/"



####################
##random data generator


#false pairs
for x in range(ln):
    p1=x
    p1l=os.listdir(pathname+dirList[p1])
    #for i in range(len(p1l)):
    for i in range(20,40):
        
        
        p1p=i

        faceImg1=cv2.imread(pathname+dirList[p1]+"/"+p1l[p1p])
        faceImg1=cv2.resize(faceImg1, (224, 224) ,interpolation = cv2.INTER_NEAREST)
        
        pairs.append(faceImg1)


del(p1l)

#pairs.shape
#pairs=pairs.astype(np.float16)

from tensorflow.keras.applications import VGG16,EfficientNetB2
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPool2D,Lambda, Conv2DTranspose, Flatten,Dense,Subtract,Input,UpSampling2D,LeakyReLU,BatchNormalization




###############################################################
decoder = Sequential()
decoder.add(Conv2D(filters=3,kernel_size=(12,12),input_shape=(224,224,3),padding="same", activation=LeakyReLU()))

decoder.add(Conv2D(filters=3,kernel_size=(12,12),padding="same", activation=LeakyReLU()))

decoder.add(BatchNormalization())
decoder.add(Conv2D(filters=64,kernel_size=(12,12),padding="same", activation=LeakyReLU()))
decoder.add(Conv2D(filters=64,kernel_size=(12,12),padding="same", activation=LeakyReLU()))


decoder.add(BatchNormalization())
decoder.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

decoder.add(Conv2D(filters=128, kernel_size=(10,10), padding="same", activation=LeakyReLU()))
decoder.add(Conv2D(filters=128, kernel_size=(10,10), padding="same", activation=LeakyReLU()))


decoder.add(BatchNormalization())
decoder.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
decoder.add(Conv2D(filters=256, kernel_size=(8,8), padding="same", activation=LeakyReLU()))
decoder.add(Conv2D(filters=256, kernel_size=(8,8), padding="same", activation=LeakyReLU()))


decoder.add(BatchNormalization())
decoder.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
decoder.add(Conv2D(filters=512, kernel_size=(6,6), padding="same", activation=LeakyReLU()))
decoder.add(Conv2D(filters=512, kernel_size=(6,6), padding="same", activation=LeakyReLU()))


decoder.add(BatchNormalization())
decoder.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
decoder.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation=LeakyReLU()))
decoder.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation=LeakyReLU()))


decoder.add(BatchNormalization())

decoder.summary()

###############################################

generator=Sequential()
generator.add(Conv2DTranspose(512, (4, 4),input_shape=(14,14,512), padding="same",activation=LeakyReLU()))
generator.add(Conv2DTranspose(512, (4, 4),input_shape=(14,14,512), padding="same",activation=LeakyReLU()))


generator.add(BatchNormalization())

generator.add(Conv2DTranspose(512, (6, 6), strides=2, padding="same", activation=LeakyReLU()))
generator.add(Conv2DTranspose(512, (6, 6), padding="same", activation=LeakyReLU()))


generator.add(BatchNormalization())

generator.add(Conv2DTranspose(256, (8, 8), strides=2, padding="same", activation=LeakyReLU()))
generator.add(Conv2DTranspose(256, (8, 8),  padding="same", activation=LeakyReLU()))


generator.add(BatchNormalization())

generator.add(Conv2DTranspose(128, (10, 10), strides=2, padding="same", activation=LeakyReLU()))
generator.add(Conv2DTranspose(128, (10, 10),  padding="same", activation=LeakyReLU()))


generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, (12, 12), strides=2, padding="same", activation=LeakyReLU()))
generator.add(Conv2DTranspose(filters=64, kernel_size=(12,12), padding="same", activation=LeakyReLU()))

generator.add(Conv2DTranspose(filters=3, kernel_size=(12,12), padding="same", activation=LeakyReLU()))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(filters=3,kernel_size=(12, 12), padding="same",activation=LeakyReLU()))
generator.summary()


################################

#    (NOT USED )()BAD RESULTS

import tensorflow as tf 

def custom_loss(y_true, y_pred):
            
    # calculate loss, using y_pred
        
    return tf.image.psnr(a=y_true, b=y_pred, max_val=255)

#################################


#"mean_squared_error"
model=Sequential()
model.add(decoder)
model.add(generator)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
   optimizer=optimizer, loss="mean_squared_error", metrics=['accuracy'])
model.fit(x=pairs,y=pairs,epochs=14)


##############################
#CHECK RESULT 
img=pairs[100]
prd=model.predict(x=np.array([img]))
plt.imshow(prd[0].astype(np.uint8))


plt.imshow(img.astype(np.uint8))



model.save("model.h5") 


from keras.models import load_model
model = load_model('model.h5')






