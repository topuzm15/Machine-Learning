import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('/home/limon/workplace/Makine Öğrenmesi/data/Mnist', one_hot=True, reshape=False)

#Todo tanımlamaların yapılması
x = tf.placeholder(tf.float32, [None, 28, 28, 1])#Bineary image bu yüzden 1 arrayi düz almıyorum 28 28 resim şeklinde aldım
y_true = tf.placeholder(tf.float32, [None, 10])
learning_rate = 5e-4
# Todo kullanılacak layer adedince layer boyutları tanımlanır
filter1 = 16
filter2 = 32
# Not: Filtre sayısını eski örnekte hidden layer şeklinde düşünebiliriz.

#Todo filtreler için weight ve biasın tanımlanması
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, filter1], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[filter1]))
w2 = tf.Variable(tf.truncated_normal([5, 5, filter1, filter2], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[filter2]))


#Todo fully connected layer için 3 layerli bir dizayn yapıyorum bu sebeble 2 w ve b değerleri tanımlıyorum
w3 = tf.Variable(tf.truncated_normal([7*7*filter2, 256], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[256]))
w4 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

#Todo filtreleme işlemini gerçekleştirelim
#                                          [batch,x ilerleme,y ilerleme, derinlik
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')) #[28,28,16] elde edtik
# Data üzerinde daha kısa süreli bir işlem yapabilmek için pool kullanırız
y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#[14,14,16]
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME'))#[14,14,32]
y2 = tf.nn.max_pool(y2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#[7,7,8]

#Todo filtre uygulanmış bir dizi resmi neural networkumuza sokalım bunun için önce vektör yap
# row kısmını bilmediğim için -1 koydum normal hali kalacak yani
flattened = tf.reshape(y2, shape=[-1, 7*7*32])
y3 = tf.nn.relu(tf.matmul(flattened, w3) + b3)

#son layerimize erişelim
y4 = tf.nn.softmax(tf.matmul(y3, w4) + b4)
#karşılaştırma kısmı
xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y4)
loss = tf.reduce_mean(xent)


correct_predict = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true,1))
percent = tf.reduce_mean(tf.cast(correct_predict, tf.float32)) * 100

#TODO b ve w değerlerini optimize etme
optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128

arrOf_loss = []

#Todo iterasyonları yapılacağı fonksiyonlar
def train(iteration):
    for i in range(iteration):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch, y_true:y_batch}
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


feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
#Todo test işlemi
def test():
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



train(10000)
test()

#Todo loss değerini görüp learning rate hakkında daha iyi yorum yapabilmek için grafiğe dökelim
plt.plot(arrOf_loss, 'r')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Loss Graph')
plt.show()

plot_example_errors()








