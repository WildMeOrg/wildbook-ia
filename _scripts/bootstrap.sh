# DEFINE INSTALLATION COMMAND
portable_install()
{
    sudo apt-get -y install $1
}

export CODE_DIR=~/code

portable_install git

# TODO: PORTS

mkdir $CODE_DIR
cd $CODE_DIR

git clone https://github.com/Erotemic/ibeis.git
cd ibeis
