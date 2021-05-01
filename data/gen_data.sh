## Download and unzip Clothing1M data set
# Generate two random Clothing1M noisy subsets for training
python3 clothing1M_rand_subset.py --name noisy_rand_subtrain1 --data-dir ./Clothing1M/ --seed 123

python3 clothing1M_rand_subset.py --name noisy_rand_subtrain2 --data-dir ./Clothing1M/ --seed 321


## Download Animal-10N from 
"https://dm.kaist.ac.kr/datasets/animal-10n/"


## You can also experiment on both CIFAR-10 and CIFAR-100
# Download CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
mv cifar-10-python.tar.gz CIFAR-10.tar.gz
tar -xvzf CIFAR-10.tar.gz
rm CIFAR-10.tar.gz

# Download CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
mv cifar-100-python.tar.gz CIFAR-100.tar.gz
tar -xvzf CIFAR-100.tar.gz
rm CIFAR-100.tar.gz