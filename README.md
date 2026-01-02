<h2 align="center">Enhancing Sample Selection Against Label Noise \\ by Cutting Mislabeled Easy Examples</h2>
<p align="center"><b>NeurIPS 2025</b> | <a href="https://openreview.net/pdf?id=OfIUAlo2hJ">[Paper]</a>
<p align="center"> <a href="https://suqinyuan.github.io">Suqin Yuan</a>, <a href="https://lfeng1995.github.io">Lei Feng</a>, <a href="https://bhanml.github.io">Bo Han</a>, <a href="https://tongliang-liu.github.io">Tongliang Liu</a> </p>

Contact: Suqin Yuan (suqinyuan.cs@gmail.com).

### TL;DR
In this paper, we demonstrate that mislabeled examples correctly predicted by the model early in the training process are particularly harmful to model performance. We refer to these examples as Mislabeled Easy Examples (MEEs). 

### BibTeX
```bibtex
@inproceedings{
yuan2025enhancing,
title={Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples},
author={Suqin Yuan and Lei Feng and Bo Han and Tongliang Liu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}
```

### Experiments

You should put the [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html) in the folder `.\cifar-10` and `.\cifar-100` if you have downloaded them.

We provide the running scripts for **CIFAR-10** and **CIFAR-100** datasets under **Symmetric** and **Instance** noise with noise rates of **20%** and **40%**.

#### CIFAR-10

Noise Rate: 40%
```bash
# Symmetric Noise
python3 main.py --dataset cifar10 --model_type resnet18 --r 0.4 --remove_rate_3 0.85 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode sym

# Instance Noise
python3 main.py --dataset cifar10 --model_type resnet18 --r 0.4 --remove_rate_3 0.85 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode instance
```

Noise Rate: 20%
```bash
# Symmetric Noise
python3 main.py --dataset cifar10 --model_type resnet18 --r 0.2 --remove_rate_3 0.925 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode sym

# Instance Noise
python3 main.py --dataset cifar10 --model_type resnet18 --r 0.2 --remove_rate_3 0.925 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode instance
```

#### CIFAR-100

Noise Rate: 40%
```bash
# Symmetric Noise
python3 main.py --dataset cifar100 --model_type resnet34 --r 0.4 --remove_rate_3 0.84 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode sym

# Instance Noise
python3 main.py --dataset cifar100 --model_type resnet34 --r 0.4 --remove_rate_3 0.84 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode instance
```

Noise Rate: 20%
```bash
# Symmetric Noise
python3 main.py --dataset cifar100 --model_type resnet34 --r 0.2 --remove_rate_3 0.925 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode sym

# Instance Noise
python3 main.py --dataset cifar100 --model_type resnet34 --r 0.2 --remove_rate_3 0.925 --i_rate_3 3 --newremove_rate 3000 --early_cutting_rate 1.5 --noise_mode instance
```
