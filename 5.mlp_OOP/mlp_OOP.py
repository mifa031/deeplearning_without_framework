import numpy as np
import time
import os
import csv
import copy    # chap 9
import wave    # chap 11
import cv2     # chap 12
import matplotlib.pyplot as plt

from PIL import Image
from IPython.core.display import HTML # chap 14

np.random.seed(1234)
def randomize(): np.random.seed(time.time())

class Model(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False
        if not hasattr(self, 'rand_std'): self.rand_std = 0.03
        
    def __str__(self):
        return '{}/{}'.format(self.name, self.dataset)
    
    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001,
                 report=0, show_cnt=3):
        self.train(epoch_count, batch_size, learning_rate, report)
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)


class MlpModel(Model):
    def __init__(self, name, dataset, hconfigs):
        super(MlpModel, self).__init__(name, dataset)
        self.init_parameters(hconfigs)

    def mlp_init_parameters(self, hconfigs):
        self.hconfigs = hconfigs
        self.pm_hiddens = []

        prev_shape = self.dataset.input_shape

        for hconfig in hconfigs:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            self.pm_hiddens.append(pm_hidden)

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)
        
    def mlp_alloc_layer_param(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig

        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w':weight, 'b':bias}, output_cnt
    
    def mlp_alloc_param_pair(self, shape):
        weight = np.random.normal(0, self.rand_std, shape)
        bias = np.zeros([shape[-1]])
        return weight, bias

    init_parameters = mlp_init_parameters
    alloc_layer_param = mlp_alloc_layer_param
    alloc_param_pair = mlp_alloc_param_pair

    def mlp_model_train(self, epoch_count=10, batch_size=10, \
                        learning_rate=0.001, report=0):
        self.learning_rate = learning_rate
        
        batch_count = int(self.dataset.train_count / batch_size)
        time1 = time2 = int(time.time())
        if report != 0:
            print('Model {} train started:'.format(self.name))

        for epoch in range(epoch_count):
            costs = []
            accs = []
            self.dataset.shuffle_train_data(batch_size*batch_count)
            for n in range(batch_count):
                trX, trY = self.dataset.get_train_data(batch_size, n)
                cost, acc = self.train_step(trX, trY)
                costs.append(cost)
                accs.append(acc)

            if report > 0 and (epoch+1) % report == 0:
                vaX, vaY = self.dataset.get_validate_data(100)
                acc = self.eval_accuracy(vaX, vaY)
                time3 = int(time.time())
                tm1, tm2 = time3-time2, time3-time1
                self.dataset.train_prt_result(epoch+1, costs, accs, acc, tm1, tm2)
                time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs:'.format(self.name, tm_total))  
    train = mlp_model_train

    def mlp_model_test(self):
        teX, teY = self.dataset.get_test_data()
        time1 = int(time.time())
        acc = self.eval_accuracy(teX, teY)
        time2 = int(time.time())
        self.dataset.test_prt_result(self.name, acc, time2-time1)
    test = mlp_model_test

    def mlp_model_visualize(self, num):
        print('Model {} Visualization'.format(self.name))
        deX, deY = self.dataset.get_visualize_data(num)
        est = self.get_estimate(deX)
        self.dataset.visualize(deX, est, deY)
    visualize = mlp_model_visualize

    def mlp_train_step(self, x, y):
        self.is_training = True

        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(x, y, output)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        self.is_training = False

        return loss, accuracy
    train_step = mlp_train_step

    def mlp_forward_neuralnet(self, x):
        hidden = x
        aux_layers = []

        for n, hconfig in enumerate(self.hconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
            aux_layers.append(aux)

        output, aux_out = self.forward_layer(hidden, None, self.pm_output)

        return output, [aux_out, aux_layers]

    def mlp_backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux

        G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

        for n in reversed(range(len(self.hconfigs))):
            hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden
    forward_neuralnet = mlp_forward_neuralnet
    backprop_neuralnet = mlp_backprop_neuralnet

    def mlp_forward_layer(self, x, hconfig, pm):
        y = np.matmul(x, pm['w']) + pm['b']
        if hconfig is not None: y = relu(y)
        return y, [x,y]

    def mlp_backprop_layer(self, G_y, hconfig, pm, aux):
        x, y = aux
        if hconfig is not None: G_y = relu_derv(y) * G_y

        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)

        pm['w'] -= self.learning_rate * G_weight
        pm['b'] -= self.learning_rate * G_bias

        return G_input
    forward_layer = mlp_forward_layer
    backprop_layer = mlp_backprop_layer

    def mlp_forward_postproc(self, output, y):
        loss, aux_loss = self.dataset.forward_postproc(output, y)
        extra, aux_extra = self.forward_extra_cost(y)
        return loss + extra, [aux_loss, aux_extra]

    def mlp_forward_extra_cost(self, y):
        return 0, None
    forward_postproc = mlp_forward_postproc
    forward_extra_cost = mlp_forward_extra_cost

    def mlp_backprop_postproc(self, G_loss, aux):
        aux_loss, aux_extra = aux
        self.backprop_extra_cost(G_loss, aux_extra)
        G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
        return G_output

    def mlp_backprop_extra_cost(self, G_loss, aux):
        pass
    backprop_postproc = mlp_backprop_postproc
    backprop_extra_cost = mlp_backprop_extra_cost

    def mlp_eval_accuracy(self, x, y, output=None):
        if output is None:
            output, _ = self.forward_neuralnet(x)
        accuracy = self.dataset.eval_accuracy(x,y,output)
        return accuracy
    eval_accuracy = mlp_eval_accuracy

    def mlp_get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.dataset.get_estimate(output)
        return estimate
    get_estimate = mlp_get_estimate


class Dataset(object):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name, self.mode,\
                                         len(self.tr_xs),len(self.te_xs),len(self,va_xs))

    @property
    def train_count(self):
        return len(self.tr_xs)

    def dataset_get_train_data(self, batch_size, nth):
        from_idx = nth * batch_size
        to_idx = (nth+1) * batch_size

        tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

        return tr_X, tr_Y

    def dataset_shuffle_train_data(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)
    get_train_data = dataset_get_train_data
    shuffle_train_data = dataset_shuffle_train_data

    def dataset_get_test_data(self):
        return self.te_xs, self.te_ys
    get_test_data = dataset_get_test_data

    def dataset_get_validate_data(self, count):
        self.va_indices = np.arange(len(self.va_xs))
        np.random.shuffle(self.va_indices)

        va_X = self.va_xs[self.va_indices[0:count]]
        va_Y = self.va_ys[self.va_indices[0:count]]

        return va_X, va_Y
    get_validate_data = dataset_get_validate_data
    get_visualize_data = dataset_get_validate_data

    def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        data_count = len(xs)

        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        self.tr_xs = xs[indices[tr_from:tr_to]]
        self.tr_ys = ys[indices[tr_from:tr_to]]
        self.va_xs = xs[indices[va_from:va_to]]
        self.va_ys = ys[indices[va_from:va_to]]
        self.te_xs = xs[indices[te_from:te_to]]
        self.te_ys = ys[indices[te_from:te_to]]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

        return indices[tr_from:tr_to], indices[va_from:va_to], indices[te_from:te_to]
    shuffle_data = dataset_shuffle_data

    def dataset_forward_postproc(self, output, y, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            diff = output - y
            square = np.square(diff)
            loss = np.mean(square)
            aux = diff
        elif mode == 'binary':
            entropy = sigmoid_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [y, output]
        elif mode == 'select':
            entropy = softmax_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [output, y, entropy]

        return loss, aux
    forward_postproc = dataset_forward_postproc

    def dataset_backprop_postproc(self, G_loss, aux, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            diff = aux
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2*diff
            g_diff_output = 1

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output= g_diff_output * G_diff
        elif mode == 'binary':
            y, output = aux
            shape = output.shape

            g_loss_entropy = np.ones(shape) / np.prod(shape)
            g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy
        elif mode == 'select':
            output, y, entropy = aux

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy

        return G_output
    backprop_postproc = dataset_backprop_postproc

    def dataset_eval_accuracy(self, x, y, output, mode=None):
        if mode is None: mode = self.mode
            
        if mode == 'regression':
            mse = np.mean(np.square(output - y))
            accuracy = 1 - np.sqrt(mse) / np.mean(y)
        elif mode == 'binary':
            estimate = np.greater(output, 0)
            answer = np.equal(y, 1.0)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
        elif mode == 'select':
            estimate = np.argmax(output, axis=1)
            answer = np.argmax(y, axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
            
        return accuracy

    eval_accuracy = dataset_eval_accuracy

    def dataset_get_estimate(self, output, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            estimate = output
        elif mode == 'binary':
            estimate = sigmoid(output)
        elif mode == 'select':
            estimate = softmax(output)

        return estimate
    get_estimate = dataset_get_estimate

    def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'. \
              format(epoch, np.mean(costs), np.mean(accs), acc, time1, time2))
    def dataset_test_prt_result(self, name, acc, time):
        print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'. \
              format(name, acc, time))
    train_prt_result = dataset_train_prt_result
    test_prt_result = dataset_test_prt_result


class AbaloneDataset(Dataset):
    def __init__(self):
        super(AbaloneDataset, self).__init__('abalone', 'regression')

        rows, _ = load_csv('./abalone.csv')

        xs = np.zeros([len(rows), 10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == 'I':xs[n,0] = 1
            if row[0] == 'M':xs[n,1] = 1
            if row[0] == 'F':xs[n,2] = 1
            xs[n,3:] = row[1:-1]
            ys[n,:] = row[-1:]

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, '%4.2f')
            print('{} => 추정 {:4.1f} : 정답 {:4.1f}'.\
                  format(xstr, est[0], ans[0]))

class PulsarDataset(Dataset):
    def __init__(self):
        super(PulsarDataset, self).__init__('pulsar', 'binary')
    
        rows, _ = load_csv('./pulsar_stars.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-1], data[:,-1:], 0.8)
        self.target_names = ['별', '펄서']
        
    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, '%5.1f', 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = 'O'
            if estr != astr: rstr = 'X'
            print('{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}'. \
                  format(xstr, estr, est[0], astr, rstr))
        
class SteelDataset(Dataset):
    def __init__(self):
        super(SteelDataset, self).__init__('steel', 'select')
    
        rows, headers = load_csv('./faults.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-7], data[:,-7:], 0.8)
        
        self.target_names = headers[-7:]
        
    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)

class PulsarSelectDataset(Dataset):
    def __init__(self):
        super(PulsarSelectDataset, self).__init__('pulsarselect', 'select')
    
        rows, _ = load_csv('./pulsar_stars.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-1], onehot(data[:,-1], 2), 0.8)
        self.target_names = ['별', '펄서']
        
    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)

class FlowersDataset(Dataset):
    pass

def flowers_init(self, resolution=[100,100], input_shape=[-1]):
    super(FlowersDataset, self).__init__('flowers', 'select')
    
    path = './flowers-recognition/flowers'
    self.target_names = list_dir(path)

    images = []
    idxs = []
    
    for dx, dname in enumerate(self.target_names):
        subpath = path + '/' + dname
        filenames = list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != '.jpg':
                continue
            imagepath = os.path.join(subpath, fname)
            pixels = load_image_pixels(imagepath, resolution, input_shape)
            images.append(pixels)
            idxs.append(dx)

    self.image_shape = resolution + [3]

    xs = np.asarray(images, np.float32)
    ys = onehot(idxs, len(self.target_names))

    self.shuffle_data(xs, ys, 0.8)

FlowersDataset.__init__ = flowers_init

def flowers_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.image_shape)
    show_select_results(estimates, answers, self.target_names)

FlowersDataset.visualize = flowers_visualize

def relu(x):
    return np.maximum(x, 0)
def relu_derv(y):
    return np.sign(y)
def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))
def sigmoid_derv(y):
    return y * (1 - y)
def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))
def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)
def tanh(x):
    return 2 * sigmoid(2*x) - 1
def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)
def softmax(x):
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()
    return probs
def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs+1.0e-10), axis=1)
def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels

def load_csv(path, skip_header=True):
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        headers = None
        if skip_header: headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)
            
    return rows, headers
def onehot(xs, cnt):
    return np.eye(cnt)[np.array(xs).astype(int)]

def vector_to_str(x, fmt='%.2f', max_cnt=0):
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt]*len(x)) % tuple(x) + ']'
    v = x[0:max_cnt]
    return '[' + ','.join([fmt]*len(v)) % tuple(v) + ',...]'
def load_image_pixels(imagepath, resolution, input_shape):
    img = Image.open(imagepath)
    resized = img.resize(resolution)
    return np.array(resized).reshape(input_shape)

def draw_images_horz(xs, image_shape=None):
    show_cnt = len(xs)
    fig, axes = plt.subplots(1, show_cnt, figsize=(5,5))
    for n in range(show_cnt):
        img = xs[n]
        if image_shape:
            x3d = img.reshape(image_shape)
            img = Image.fromarray(np.uint8(x3d))
        axes[n].imshow(img)
        axes[n].axis('off')
    plt.draw()
    plt.show()
def show_select_results(est, ans, target_names, max_cnt=0):
    for n in range(len(est)):
        pstr = vector_to_str(100*est[n], '%2.0f', max_cnt)
        estr = target_names[np.argmax(est[n])]
        astr = target_names[np.argmax(ans[n])]
        rstr = 'O'
        if estr != astr: rstr = 'X'
        print('추정확률분포 {} => 추정 {} : 정답 {} => {}'. \
              format(pstr, estr, astr, rstr))
def list_dir(path):
    filenames = os.listdir(path)
    filenames.sort()
    return filenames


####################### test ##########################

ad = AbaloneDataset()
am = MlpModel('abalone_model', ad, [])
am.exec_all(epoch_count=10, report=2)

pd = PulsarDataset()
pm = MlpModel('pulsar_model', pd, [4])
pm.exec_all()
pm.visualize(5)

sd = SteelDataset()
sm = MlpModel('steel_model', sd, [12,7])
sm.exec_all(epoch_count=50, report=10)

psd = PulsarSelectDataset()
psm = MlpModel('pulsar_select_model', psd, [4])
psm.exec_all()

fd = FlowersDataset()
fm = MlpModel('flowers_model_1', fd, [10])
fm.exec_all(epoch_count=10, report=2)

fm2 = MlpModel('flowers_model_2', fd, [30, 10])
fm2.exec_all(epoch_count=10, report=2)
