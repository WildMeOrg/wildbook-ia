ibeis
=====

image based ecological information system


Environment Setup: 

# NAVIGATE TO YOUR CODE DIRECTORY
cd %USERPROFILE%\code
cd ~/code

# CLONE THESE REPOS
git clone https://github.com:Erotemic/utool.git
git clone https://github.com:Erotemic/vtool.git
git clone https://github.com/Erotemic/hesaff.git
git clone https://github.com/Erotemic/ibeis.git
# Set the previous repos up for development by running this
# command in each directory
sudo python setup.py develop

# Clone these repos
git clone https://github.com/Erotemic/opencv.git
git clone https://github.com/Erotemic/flann.git
git clone https://github.com/bluemellophone/pyrf.git
# Use the build scripts in their for either unix or mingw
# you dont need to build detecttools
git clone https://github.com/bluemellophone/detecttools.git
<!--git clone https://github.com/bluemellophone/IBEIS2014.git-->
