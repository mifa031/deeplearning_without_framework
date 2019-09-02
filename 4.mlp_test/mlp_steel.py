import numpy as np
import csv
import time

np.random.seed(1234)
def randomize(): np.random.seed(time.time())

RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001

def init_model_hidden1():
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt

    pm_hidden = alloc_param_pair([input_cnt, hidden_cnt])
    pm_output = alloc_param_pair([hidden_cnt, output_cnt])

def alloc_param_pair(shape):
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])
    return {'w':weight, 'b':bias}

def forward_neuralnet_hidden1(x):
    global pm_output, pm_hidden

    hidden = relu(np.matmul(x, pm_hidden['w']) + pm_hidden['b'])
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    return output, [x, hidden]

def backprop_neuralnet_hidden1(G_output, aux):
    global pm_output, pm_hidden

    x, hidden = aux

    g_output_w_out = hidden.transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out

    G_hidden = G_hidden * relu_derv(hidden)

    g_hidden_w_hid = x.transpose()
    G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
    G_b_hid = np.sum(G_hidden, axis=0)

    pm_hidden['w'] -= LEARNING_RATE * G_w_hid
    pm_hidden['b'] -= LEARNING_RATE * G_b_hid

def relu_derv(y):
    return np.sign(y)

def init_model_hiddens():
    global pm_output, pm_hiddens, input_cnt, output_cnt, hidden_config

    pm_hiddens = []
    prev_cnt = input_cnt

    for hidden_cnt in hidden_config:
        pm_hiddens.append(alloc_param_pair([prev_cnt, hidden_cnt]))
        prev_cnt = hidden_cnt

    pm_output = alloc_param_pair([prev_cnt, output_cnt])

def forward_neuralnet_hiddens(x):
    global pm_output, pm_hiddens

    hidden = x
    hiddens = [x]

    for pm_hidden in pm_hiddens:
        hidden = relu(np.matmul(hidden, pm_hidden['w']) + pm_hidden['b'])
        hiddens.append(hidden)

    output = np.matmul(hidden, pm_output['w']) + pm_output['b']

    return output, hiddens

def backprop_neuralnet_hiddens(G_output, aux):
    global pm_output, pm_hiddens

    hiddens = aux

    g_output_w_out = hiddens[-1].transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out

    for n in reversed(range(len(pm_hiddens))):
        G_hidden = G_hidden * relu_derv(hiddens[n+1])

        g_hidden_w_hid = hiddens[n].transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)

        g_hidden_hidden = pm_hiddens[n]['w'].transpose()
        G_hidden = np.matmul(G_hidden, g_hidden_hidden)

        pm_hiddens[n]['w'] -= LEARNING_RATE * G_w_hid
        pm_hiddens[n]['b'] -= LEARNING_RATE * G_b_hid

global hidden_config
def init_model():
    if hidden_config is not None:
        print('은닉계층 {}개를 갖는 다층 퍼셉트론 작동'.
              format(len(hidden_config)))
        init_model_hiddens()
    else:
        print('은닉계층 1개를 갖는 다층 퍼셉트론 작동')
        init_model_hidden1()

def forward_neuralnet(x):
    if hidden_config is not None:
        return forward_neuralnet_hiddens(x)
    else:
        return forward_neuralnet_hidden1(x)

def backprop_neuralnet(G_output, hiddens):
    if hidden_config is not None:
        backprop_neuralnet_hiddens(G_output, hiddens)
    else:
        backprop_neuralnet_hidden1(G_output, hiddens)

def set_hidden(info):
    global hidden_cnt, hidden_config
    if isinstance(info, int):
        hidden_cnt = info
        hidden_config = None
    else:
        hidden_config = info

def steel_exec(epoch_count=10, mb_size=10, report=1):
    load_steel_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_steel_dataset():
    with open('./faults.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 27, 7
    data = np.asarray(rows, dtype='float32')

def forward_postproc(output, y):
    entropy = softmax_cross_entropy_with_logits(y, output)
    loss = np.mean(entropy)
    return loss, [y, output, entropy]

def backprop_postproc(G_loss, aux):
	y, output, entropy = aux

	g_loss_entropy = 1.0 / np.prod(entropy.shape)
	g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

	G_entropy = g_loss_entropy * G_loss
	G_output = g_entropy_output * G_entropy

	return G_output

def eval_accuracy(output, y):
    estimate = np.argmax(output, axis=1)
    answer = np.argmax(y, axis=1)
    correct = np.equal(estimate, answer)

    return np.mean(correct)

def softmax(x):
	max_elem = np.max(x, axis=1)
	diff = (x.transpose() - max_elem).transpose()
	exp = np.exp(diff)
	sum_exp = np.sum(exp, axis=1)
	probs = (exp.transpose() / sum_exp).transpose()
	return probs

def softmax_derv(x, y):
	mb_size, nom_size = x.shape
	derv = np.ndarray([mb_size, nom_size, nom_size])
	for n in range(mb_size):
		for i in range(nom_size):
			for j in range(nom_size):
				derv[n, i, j] = -y[n,i] * y[n,j]
			derv[n,i,i] += y[n,i]
	return derv

def softmax_cross_entropy_with_logits(labels, logits):
	probs = softmax(logits)
	return -np.sum(labels * np.log(probs+1.0e-10), axis=1)

def softmax_cross_entropy_with_logits_derv(labels, logits):
	return softmax(logits) - labels

def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])

def train_and_test(epoch_cnt, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_cnt):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:.3f}, accuracy={:.3f}/{:.3f}'\
                  .format(epoch+1, np.mean(losses), np.mean(accs), acc))
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:.3f}'.format(final_acc))

def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:,:-output_cnt], test_data[:, -output_cnt:]

def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth : mb_size*(nth+1)]]
    return train_data[:,:-output_cnt], train_data[:,-output_cnt:]

def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy

def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy

def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x

def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b

def backprop_postproc_oneline(G_loss, diff):
    return 2 * diff / np.prod(diff.shape)

set_hidden([12,6,4])
steel_exec(epoch_count=50, report=10)
