## Split into train / val / test
python preprocess_cifar10.py

python preprocess_cifar100.py


### CIFAR-10
## symmetric noise

python preprocess_sym_noise_cifar.py --noise-rate 0.2 --inDir CIFAR10/train --outDir CIFAR10/train_sn_0.2 --dataset CIFAR10

python preprocess_sym_noise_cifar.py --noise-rate 0.5 --inDir CIFAR10/train --outDir CIFAR10/train_sn_0.5 --dataset CIFAR10

python preprocess_sym_noise_cifar.py --noise-rate 0.8 --inDir CIFAR10/train --outDir CIFAR10/train_sn_0.8 --dataset CIFAR10


## asymmetric noise

python preprocess_asym_noise_cifar.py --noise-rate 0.3 --inDir CIFAR10/train --outDir CIFAR10/train_an_0.3 --dataset CIFAR10

python preprocess_asym_noise_cifar.py --noise-rate 0.5 --inDir CIFAR10/train --outDir CIFAR10/train_an_0.5 --dataset CIFAR10




### CIFAR-100

## symmetric noise

python preprocess_sym_noise_cifar.py --noise-rate 0.2 --inDir CIFAR100/train --outDir CIFAR100/train_sn_0.2 --dataset CIFAR100

python preprocess_sym_noise_cifar.py --noise-rate 0.5 --inDir CIFAR100/train --outDir CIFAR100/train_sn_0.5 --dataset CIFAR100

python preprocess_sym_noise_cifar.py --noise-rate 0.8 --inDir CIFAR100/train --outDir CIFAR100/train_sn_0.8 --dataset CIFAR100


## asymmetric noise

python preprocess_asym_noise_cifar.py --noise-rate 0.3 --inDir CIFAR100/train --outDir CIFAR100/train_an_0.3 --dataset CIFAR100

python preprocess_asym_noise_cifar.py --noise-rate 0.5 --inDir CIFAR100/train --outDir CIFAR100/train_an_0.5 --dataset CIFAR100