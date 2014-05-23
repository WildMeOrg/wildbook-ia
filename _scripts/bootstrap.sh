# DEFINE INSTALLATION COMMAND
portable_install()
{
    sudo apt-get -y install $1
}

export INSTALL_DIR=~/code

portable_install git

mkdir $INSTALL_DIR
cd $INSTALL_DIR

git clone https://github.com/Erotemic/ibeis.git
