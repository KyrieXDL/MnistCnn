import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import cv2
import minist_project.cnn_utils as cnn_utils
from minist_project.input_data import load_train_images, load_train_labels, load_test_images, load_test_labels

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [5,5,1,6], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5,5,6,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1":W1, "W2":W2}

    return parameters

# def forward_propagation(X, parameters):
#     W1 = parameters["W1"]
#     W2 = parameters["W2"]
#
#     #conv1
#     Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
#     A1 = tf.nn.relu(Z1)
#     #pool1
#     P1 = tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
#
#     #conv2
#     Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
#     A2 = tf.nn.relu(Z2)
#     #pool2
#     P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
#     #flatten
#     P = tf.contrib.layers.flatten(P2)
#
#     #fc1
#     a3 = tf.contrib.layers.fully_connected(P, 1024)
#     #fc2
#     Z5 = tf.contrib.layers.fully_connected(a3, 10, activation_fn=None)
#
#     return Z5

#LeNet5网络结构
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #conv1
    Z1 = tf.nn.conv2d(input=X,filter=W1,strides=[1,1,1,1],padding="VALID")
    A1 = tf.nn.relu(Z1)
    #pool1
    P1 = tf.nn.max_pool(value=A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #conv2
    Z2 = tf.nn.conv2d(input=P1,filter=W2,strides=[1,1,1,1],padding="VALID")
    A2 = tf.nn.relu(Z2)
    #pool2
    P2 = tf.nn.max_pool(value=A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #flatten
    P = tf.contrib.layers.flatten(P2)

    #fc1
    f1 = tf.contrib.layers.fully_connected(P, 120)
    #fc2
    f2 = tf.contrib.layers.fully_connected(f1, 84)
    #fc3
    Z = tf.contrib.layers.fully_connected(f2, 10, activation_fn=None)

    return Z

def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,num_epochs=150,minibatch_size=64,print_cost=True,isPlot=True):
    tf.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape #获取数据集的维度
    n_y = Y_train.shape[1]
    costs = []  #用于存放我们每次迭代的代价

    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()
    Z = forward_propagation(X,parameters)
    cost = compute_cost(Z, Y)
    #这里使用了Adam的优化器，Adam优化了我们minibatch梯度下降算法，使下降更快
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  #创建saver用于保存训练后的模型
    total_time = 0  #记录每5次迭代的总时间
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            start_time = time.clock()
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            #分割数据集
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                #将我们的每个batch的数据传入我们的网络中，然后进行梯度下降
                _, temp_cost = sess.run([optimizer, cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
            end_time = time.clock()
            minium = end_time - start_time
            total_time += minium
            if print_cost:
                if epoch % 10 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost) + " ; 每一个epoch花费时间：" + str(minium) + " 秒，10个epoch总的时间：" + str(total_time))
                    total_time = 0

            if epoch % 10 == 0:
                costs.append(minibatch_cost)
		#保存模型
        saver.save(sess, "ModelLeNet5/MnistModel")
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iterations (per tens)")
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
		#tf.argmax的第二个参数等于1表示我们输出Z中每行最大值的索引
        predict_op = tf.argmax(Z, 1)
        #tf.equal比较两个矩阵或向量对应元素是否相等，相等就为True，不等就为False
        corrent_prediction = tf.equal(predict_op, tf.argmax(Y,1))
		#tf.cast是进行数据转换，这里是bool型转为float，True就为1.0，False就是0.0
		#tf.reduce_mean就是求均值
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
		#这里同样将我们的数据传入我们的tensor张量中
        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

        print("训练集准确度：" + str(train_accuracy))
        print("测试及准确度：" + str(test_accuracy))

    return (train_accuracy, test_accuracy, parameters)

def load_model(x):
    # plt.imshow(x)
    # plt.show()
    x = x.reshape(1,28,28,1)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("ModelLeNet5/MnistModel.meta")  # 加载图
        saver.restore(sess, tf.train.latest_checkpoint("ModelLeNet5/"))  # 加载模型
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('Placeholder:0')  # 获取保存模型中的输入变量的tensor
        Z = graph.get_tensor_by_name("fully_connected_2/BiasAdd:0")  # 获取softmax层的输入值
        res = tf.nn.softmax(Z)  # 执行softmax激活函数
        preds = sess.run(res, feed_dict={X: x})  # 进行预测
        print(preds)
        print("预测结果 ", np.argmax(preds))

        return np.squeeze(preds)


if __name__ == "__main__":
    train_x = load_train_images()
    train_y = load_train_labels()
    test_x = load_test_images()
    test_y = load_test_labels()
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
    train_y = train_y.reshape(len(train_y), 1).astype(int)
    test_y = test_y.reshape(len(test_y), 1).astype(int)
    train_y = cnn_utils.convert_to_one_hot(train_y, 10)
    test_y = cnn_utils.convert_to_one_hot(test_y, 10)

    print("训练集x：", train_x.shape)
    print("训练集y：", train_y.shape)
    print("测试机x：", test_x.shape)
    print("测试机y：", test_y.shape)
    # # 训练模型
    # tf.reset_default_graph()
    # np.random.seed(1)
    # _, _, parameters = model(train_x, train_y, test_x, test_y, num_epochs=30)

    # 加载模型
    load_model(test_x[100])