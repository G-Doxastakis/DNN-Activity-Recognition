import numpy as np
import pandas
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load(filename, shuffle=False, shuffle_len=1000):
    df = pandas.read_csv(filename)
    #print(df['User'].astype('category').cat.categories)
    class_names = df['gt'].astype('category').cat.categories
    print(class_names)
    df['User'] = df['User'].astype('category').cat.codes
    df['gt'] = df['gt'].astype('category').cat.codes #.fillna(4)  # .fillna(6) TODO: Find null class
    df = df.fillna(0)
    data = df.values
    if shuffle:
        data = data[0:(len(data)//shuffle_len)*shuffle_len, :]
        data = np.random.permutation(data.reshape((len(data)//shuffle_len, -1, 5))).reshape(-1, 5)
    return data, class_names


def generator(data, idx_range, seq_len):
    while True:
        for i in idx_range:
            yield (data[i-seq_len:i, 0:3].reshape((-1, seq_len, 3)),
                   to_categorical(data[i, 4], num_classes=6).reshape(-1, 6))


def randgenerator(data, idx_range, seq_len):
    while True:
        i = np.asscalar(np.random.randint(idx_range.start, idx_range.stop, size=1))
        yield (data[i-seq_len:i, 0:3].reshape((-1, seq_len, 3)),
               to_categorical(data[i, 4], num_classes=6).reshape(-1, 6))


def exportmodel(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
        False, 'out/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


def confusionMatrix(model, generator, nsamples, class_names):
    data = itertools.islice(generator, nsamples)
    x_test = []
    y_test = []
    for i in data:
        x_test.append(i[0])
        y_test.append(i[1])

    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    y_pred = model.predict(x_test)

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_test)
    print(y_pred)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

