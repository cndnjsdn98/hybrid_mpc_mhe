# Hybrid-MPC-MHE
Hybrid Model driven MPC & MHE

export PYTHONPATH=$PWD/:$PYTHONPATH

# Install Requirements

## Setup Python Virtual Environment
1. Install pip for Python3
```
sudo apt update
sudo apt install -y python3-pip
python3.9 -m pip install --upgrade pip
```

2. Setup Python virtual environment for the packages. 
```
python3.9 -m pip install virtualenv

cd <PATH_TO_PROJECT_DIRECTORY>
virtualenv node --python=/usr/bin/python3.9
source ./node/bin/activate
```

## Install Python Packages
1. Intall packages from requirements.txt
```
cd <PATH_TO_PROJECT_DIRECTORY>
pip install -r requirements.txt
```

2. Install CPU Only PyTorch
This package utilizes the CPU only version of L4Casadi package that is utilized to integrate PyTorch model to CasADi. If the user wishes to use GPU also, then follow install steps [here](https://github.com/Tim-Salzmann/l4casadi) to install for GPU also.

```
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu
```

3. Install [PyTorch Implementation of Differential ODE Solvers](https://github.com/rtqichen/torchdiffeq)
```
pip install torchdiffeq
```

3. Install [L4CasADi](https://github.com/Tim-Salzmann/l4casadi)
```
pip install l4casadi --no-build-isolation
```
