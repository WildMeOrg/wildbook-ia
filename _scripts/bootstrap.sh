# DEFINE INSTALLATION COMMAND

update_package_manager()
{
    sudo apt-get update
}

portable_install()
{
    sudo apt-get -y install $1
}

python_install()
{
    sudo pip install $1 --upgrade
}

export CODE_DIR=~/code

update_package_manager()

portable_install cmake
portable_install gcc
portable_install g++
portable_install git
portable_install python-pip
portable_install python-dev
portable_install python-setuptools

python_install numpy
python_install scipy
python_install ipython

# PyQt4
sudo apt-get install -y python-qt4

# DO OPENCV ON MAC

# TODO: PORTS

mkdir $CODE_DIR
cd $CODE_DIR

git clone https://github.com/Erotemic/ibeis.git
cd ibeis

sudo python super_setup.py --build --develop
