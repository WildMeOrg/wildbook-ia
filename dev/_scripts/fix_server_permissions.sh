cd /opt
sudo chown -R root:ibeis ibeis
sudo chmod -R g+w ibeis

sudo chown -R root:ibeis "$(git rev-parse --show-toplevel)/.git"
sudo chmod -R g+w "$(git rev-parse --show-toplevel)/.git"
sudo chmod -R g+w *
sudo chown -R root:ibeis *
$USER:$USER 
