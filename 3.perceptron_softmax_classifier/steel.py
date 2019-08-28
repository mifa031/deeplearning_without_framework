import numpy as np
import csv
import time

np.random.seed(1234)
def randomize(): np.random.seed(time.time())

RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001

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

steel_exec()
