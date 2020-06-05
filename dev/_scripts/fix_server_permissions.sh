cd /opt
sudo chown -R root:wbia wbia
sudo chmod -R g+w wbia

sudo chown -R root:wbia "$(git rev-parse --show-toplevel)/.git"
sudo chmod -R g+w "$(git rev-parse --show-toplevel)/.git"
sudo chmod -R g+w *
sudo chown -R root:wbia *
$USER:$USER
