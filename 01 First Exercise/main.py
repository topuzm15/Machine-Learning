import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#kütüphanedeki datanın okunması
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

#giriş ve sonuç verilerini ata
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

# w ve b değerlerini ata
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# işlemi gerçekleştir
logits = tf.matmul(x, w) + b

#1-0 arası normal dist. ile dağıt
y = tf.nn.softmax(logits)

#cost değerini hesapla
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

# yolladığımız datalar grub halinde olduğundan grubdaki ortalama hata miktarını hesapla
loss = tf.reduce_mean(xent)

# Tüm bunları True, False olarak bir listeye atıyoruz.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
# Yüzde hata değerini hesapla
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

# w ve b değerlerini değiştirme işlemini gerçekleştirelim
optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss) ### 0.5 öğrenme oranı

##########################################################
####    Kodun çalışması için Gerekli komutlar    #########

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Büzün datayı tek seferde göndermek yerine parça parça yolluyorum
batch_size = 128

#şimdiki 2 fonksiyonumuz iterasyon ile w ve b değerlerinin en doğru sonuçlarını bulmak ve olasılığı bastırmak için
def train(iteration):
    for i in range(iteration):

        #öncelikle resimleri ve sonuçları alalım
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        # yukarıdaki eğitim kısmına atalım
        feed_dic_train = {x:x_batch, y_true:y_batch}

        # eğitimi gerçekleştirelim
        sess.run(optimize, feed_dict=feed_dic_train)

def result():
    # test resimlerini al
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    occ = sess.run(accuracy, feed_dict=feed_dict_test)
    print('doğruluk oranı %', occ)


train(10000)
result()