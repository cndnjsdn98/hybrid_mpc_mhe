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
virtualenv hybrid --python=/usr/bin/python3.9
source ./hybrid/bin/activate
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

3. Install [PyTorch Implementation of Gaussian Process](https://gpytorch.ai/)

```
pip install gpytorch
```

5. Install [PyTorch Implementation of Differential ODE Solvers](https://github.com/rtqichen/torchdiffeq)
```
pip install torchdiffeq
```

5. Install [L4CasADi](https://github.com/Tim-Salzmann/l4casadi)
```
pip install l4casadi --no-build-isolation
```

6. Install [Acados](https://docs.acados.org/index.html)

# Setup Catkin Workspace

```
sudo apt-get install ros-${ROS_DISTRO}-mavros ros-${ROS_DISTRO}-mavros-extras ros-${ROS_DISTRO}-mavros-msgs ros-${ROS_DISTRO}-mavlink
```

```
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh
```

```
rm ./install_geographiclib_datasets.sh
```

```
vcs-import < hybrid_mpc_mhe/dependencies.yaml

git clone https://github.com/mavlink/mavlink.git --recursive
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
```
