from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/limon/workplace/Makine Öğrenmesi/data/Mnist'
                                  ,one_hot=True,reshape=False)

#oluşturacağımız deep learning modelini tanımladım
model = Sequential()

#modelin input kısmını ayarlıyorum
model.add(InputLayer(input_shape=(28, 28, 1)))

#################################################################################
#Todo conv1 için yapılan işlemler
#
#1 adet cnn ekliyorum 5 lik filtreler 1 er kayar ve 32 aded boyut aynı
#aktivasyon fonk olarak relu kullanılır ve ismi conv1
model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same'
                 , activation='relu', name='conv1'))

#max pool ile ortalama değerler elde edilip data küçültülür
model.add(MaxPool2D(pool_size=2, strides=2))
################################################################################

#################################################################################
#Todo conv1 için yapılan işlemler
#
#1 adet cnn ekliyorum 5 lik filtreler 1 er kayar ve 64 aded boyut aynı
#aktivasyon fonk olarak relu kullanılır ve ismi conv1
model.add(Conv2D(kernel_size=5,strides=1, filters=64,padding='same'
                 ,activation='relu',name='conv2'))

#max pool ile ortalama değerler elde edilip data küçültülür
model.add(MaxPool2D(pool_size=2, strides=2))
################################################################################


################################################################################
#Todo full connected layer kısmına geçiş yapyorum şimdide
#modelimizi düzleyelim önce
model.add(Flatten())

#full connected 1. layer
model.add(Dense(512, activation='relu',name='full1'))
model.add(Dense(512, activation='relu', name='full2'))
model.add(Dense(10,activation='softmax'))
###############################################################################


###############################################################################
#Todo optimizer tanımlanması
#learning rate ile optimizer tanımlanır
optimizer = Adam(lr=1e-5)
###############################################################################

##################################################################################
#Todo modelin training kısmı
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
metricts = ['accuracy']
model.fit(x=mnist.train.images,y=mnist.train.labels,epochs=3,batch_size=128)
result = model.evaluate(x=mnist.test.images,y=mnist.test.labels)
print('\naccuracy: %',result[1]*100)






























































