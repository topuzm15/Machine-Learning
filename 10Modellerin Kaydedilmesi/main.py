import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
import os


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
#save ile kacıncı iterasyonda kaldığımızıda kontrol edelim
global_step = tf.Variable(0,trainable=False)
optimize = tf.train.AdamOptimizer(0.001).minimize(loss,global_step)

sess = tf.Session()
#TODO kayıt etme işlemi
saver = tf.train.Saver()
save_dir = 'checkpoints/'

#dosyanın olup olmadığını kontrol etme işlemi
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model')


#save veya 0 dan başlama işleminin gerçekleşmesi
try:
    print('Check point yukleniyor...')
    last_chck_point = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess,save_path=last_chck_point)
    print('Ok ', last_chck_point)
except:
    print('Check point yok arkedes')
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
        #global stepide yükleyelim ve kaydedelim
        [_, train_loss,glob_step] = sess.run([optimize, loss, global_step], feed_dict=feed_dict_train)

        #100 iterasyonda bir loss durumunu göster
        if i % 100 == 0:
            train_acc = sess.run(percent,feed_dict=feed_dict_train)

            # loading oluştur
            if(i %1000 == 0):
                sharp = int(i/1000) * '#'
                sharpPercent = '|' + sharp + (9-len(sharp)) * ' ' + '|'
            arrOf_loss.append(train_loss)
            print(sharpPercent, 'predict: %', train_acc, 'loss: ', train_loss)

            if i %1000 == 0:
                saver.save(sess,save_path=save_path,global_step=global_step)
                print('Kayıt islemi tamamlandı')

feed_dict_test = {inputs: mnist.test.images, output: mnist.test.labels, pkeep:1}
#Todo test işlemi
def test():
    #Todo test işlemi sırasında tüm nöronları kullanacağımz için pkeep %100 olmalı
    feed_dict_test = {inputs: mnist.test.images, output: mnist.test.labels, pkeep:1}
    acc = sess.run(percent, feed_dict=feed_dict_test)
    print('truth value = %', acc)


def plot_images(images, cls_true, cls_pred=None):
    # assert doğru olup olmadığına dair kontrol mekanizması
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # enumerate sayesinde axes.flat kısmındaki vektörleri array numaraları ile birlikte alırıs
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y3, 1)
    correct, cls_pred = sess.run([correct_predict, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])



#train(10000)
test()

#Todo loss değerini görüp learning rate hakkında daha iyi yorum yapabilmek için grafiğe dökelim
plt.plot(arrOf_loss, 'r')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss Graph')
plt.show()

plot_example_errors()