import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
import numpy as np
from math import inf
from scipy import stats
from PIL import Image

def dataset_split(train_images, train_labels,  noise_rate=0.5, noise_type='sym', random_seed=1, num_classes=10, include_noise=False, dataset=None):
    train_labels = np.array(train_labels)
    if include_noise:
        noise_rate = noise_rate * (1 - 1 / num_classes)
        print("include_noise True, new real nosie rate:", noise_rate)

    clean_train_labels = train_labels[:, np.newaxis]

    if noise_type == 'instance':
        norm_std = 0.1
        if(len(train_images.shape) == 2):
            feature_size = train_images.shape[1]
        else:
            feature_size = 1
            for i in range(1, len(train_images.shape)):
                feature_size = int(feature_size * train_images.shape[i])

        if torch.is_tensor(train_images) is False:
            data = torch.from_numpy(train_images)
        else:
            data = train_images

        data = data.type(torch.FloatTensor)
        targets = torch.from_numpy(train_labels)
        dataset = zip(data, targets)
        noisy_labels = get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, random_seed)

    elif noise_type == 'sym':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
        # print(transition_matrix)
    elif noise_type == 'pair':
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
        # print(transition_matrix)
    elif noise_type == 'asym':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_asymmetric(
            clean_train_labels, noise=noise_rate, random_state=random_seed, nb_classes=num_classes, dataset=dataset)

    noisy_labels = noisy_labels.squeeze()
    return noisy_labels

def dataset_split_imagenet(train_data, train_labels, noise_rate=0.5, noise_type='sym', random_seed=1, num_classes=1000, include_noise=False):
    train_labels = np.array(train_labels)
    if include_noise:
        noise_rate = noise_rate * (1 - 1 / num_classes)
        print("include_noise True, new real noise rate:", noise_rate)

    clean_train_labels = train_labels[:, np.newaxis]

    if noise_type == 'sym':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(
            clean_train_labels,
            noise=noise_rate,
            random_state=random_seed,
            nb_classes=num_classes
        )
    elif noise_type == 'pair':
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(
            clean_train_labels,
            noise=noise_rate,
            random_state=random_seed,
            nb_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    noisy_labels = noisy_labels.squeeze()
    return noisy_labels


def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P

def get_instance_noisy_label(n, newdataset, labels, num_classes, feature_size, norm_std, seed):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (0.6 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)
    if torch.cuda.is_available():
        W = torch.FloatTensor(W).cuda()
    else:
        W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(newdataset):
        if torch.cuda.is_available():
            x = x.cuda()
            x = x.reshape(feature_size)
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l1 = [i for i in range(label_num)]
    new_label = [np.random.choice(l1, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    return np.array(new_label)

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):

    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print(y_train_noisy)
        y_train = y_train_noisy

    return y_train, actual_noise, P

def noisify_multiclass_asymmetric(y_train, noise, random_state=None, nb_classes=10, dataset=None):
    """添加非对称噪声。"""
    P = np.eye(nb_classes)
    n = noise
    if n > 0.0:
        if dataset == 'mnist' and nb_classes == 10:
            # MNIST 非对称噪声翻转规则
            P[7, 7], P[7, 1] = 1. - n, n   # 7 -> 1
            P[2, 2], P[2, 7] = 1. - n, n   # 2 -> 7
            P[5, 5], P[5, 6] = 1. - n, n   # 5 -> 6
            P[6, 6], P[6, 5] = 1. - n, n   # 6 -> 5
            P[3, 3], P[3, 8] = 1. - n, n   # 3 -> 8
        elif dataset == 'cifar10' and nb_classes == 10:
            # CIFAR-10 非对称噪声翻转规则
            P[9, 9], P[9, 1] = 1. - n, n   # truck -> automobile
            P[2, 2], P[2, 0] = 1. - n, n   # bird -> airplane
            P[3, 3], P[3, 5] = 1. - n, n   # cat -> dog
            P[5, 5], P[5, 3] = 1. - n, n   # dog -> cat
            P[4, 4], P[4, 7] = 1. - n, n   # deer -> horse
        elif dataset == 'cifar100' and nb_classes == 100:
            # CIFAR-100 非对称噪声翻转规则
            nb_superclasses = 20
            nb_subclasses = 5
            for i in range(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)
        else:
            # 对于其他数据集，定义默认的非对称翻转规则
            for i in range(nb_classes):
                P[i, i] = 1. - n
                P[i, (i + 1) % nb_classes] = n
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]), decimal=6)
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
    else:
        y_train_noisy = y_train.copy()
        actual_noise = 0.0
    return y_train_noisy, actual_noise, P


def build_for_cifar100(size, noise):
    """构建 CIFAR-100 非对称噪声的转移概率矩阵。"""
    assert (noise >= 0.) and (noise <= 1.)
    P = (1. - noise) * np.eye(size)
    for i in range(size - 1):
        P[i, i + 1] = noise
    P[size - 1, 0] = noise
    assert_array_almost_equal(P.sum(axis=1), np.ones(size), 1)
    return P

def multiclass_noisify(y, P, random_state=1):

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # i is np.array, such as [1]
        if not isinstance(i, np.ndarray):
            i = [i]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

