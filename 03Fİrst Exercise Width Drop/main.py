import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets('/home/limon/workplace/Makine Öğrenmesi/data/Mnist' , one_hot=True)

#TODO tanımlamalar

layer0 = 784
layer1 = 16 # hidden layer
layer2 = 16 # hidden layer
layer_O = 10

inputs = tf.placeholder(tf.float32, [None, layer0])
output = tf.placeholder(tf.float32, [None, layer_O])
# Droplanacak nodeler için placeholder atanması NOT: bu değer iteration sırasında atanacak
pkeep  = tf.placeholder(tf.float32)


w1 = tf.Variable(tf.truncated_normal([layer0, layer1], stddev= 0.1)) # stdev standart sapma manasındadır

b1 = tf.Variable(tf.constant(0.1, shape=[layer1])) # w'nin outputu olacak şekilde
w2 = tf.Variable(tf.truncated_normal([layer1, layer2], stddev= 0.1)) # stdev standart sapma manasındadır
b2 = tf.Variable(tf.constant(0.1, shape=[layer2])) # w'nin outputu olacak şekilde
w3 = tf.Variable(tf.truncated_normal([layer2, layer_O], stddev= 0.1)) # stdev standart sapma manasındadır
b3 = tf.Variable(tf.constant(0.1, shape=[layer_O])) # w'nin outputu olacak şekilde

##########################################
#####  TODO Yapılacak işlemler     #######

#TODO dipnot: aktivasyon fonksiyonları nn kütüphanesinde bulunur

##aktivasyon fonksiyonu olarak relu kullandım
y1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
y1d = tf.nn.dropout(y1,pkeep)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y2d = tf.nn.dropout(y2,pkeep)
# Daha sonra loss fonksiyonunda kullanılacağı için out labeli ayrı yazıyorum
logit = tf.matmul(y2, w3) + b3
# y3' e yani sonuc kısmında drop kuralını uygulamayacağız.
y3 = tf.nn.softmax(logit)


#TODO costun hesaplanması
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=output)
loss = tf.reduce_mean(xent)

correct_predict = tf.equal(tf.argmax(y3, 1), tf.argmax(output,1))
percent = tf.reduce_mean(tf.cast(correct_predict, tf.float32)) * 100

#TODO b ve w değerlerini optimize etme
optimize = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
arrOf_loss = []

#Todo iterasyonları yapılacağı fonksiyonlar
def train(iteration):
    for i in range(iteration):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {inputs:x_batch, output:y_batch,pkeep:0.75}
        # loss değirindeki yani toplam costdaki değişimide görmek için
        #Todo pkeep drop oranı %75 olarak atandı
        [_, train_loss] = sess.run([optimize, loss], feed_dict=feed_dict_train)

        #100 iterasyonda bir loss durumunu göster
        if i % 100 == 0:
            train_acc = sess.run(percent,feed_dict=feed_dict_train)

            # loading oluştur
            if(i %1000 == 0):
                sharp = int(i/1000) * '#'
                sharpPercent = '|' + sharp + (9-len(sharp)) * ' ' + '|'
            arrOf_loss.append(train_loss)
            print(sharpPercent, 'predict: %', train_acc, 'loss: ', train_loss)

#Todo test işlemi
def test():
    #Todo test işlemi sırasında tüm nöronları kullanacağımz için pkeep %100 olmalı
    feed_dict_test = {inputs: mnist.test.images, output: mnist.test.labels, pkeep:1}
    acc = sess.run(percent, feed_dict=feed_dict_test)
    print('truth value = %', acc)


train(10000)
test()
#Todo loss değerini görüp learning rate hakkında daha iyi yorum yapabilmek için grafiğe dökelim
plt.plot(arrOf_loss, 'r')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss Graph')
plt.show()
