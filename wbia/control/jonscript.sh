# From new terminator
# cd ~/code/wbia/wbia/control/
# ibc && sh jonscript.sh

ORIG=$PS1
TITLE="\e]2;\"WILDBOOK_TEST_TERM\"\a"
PS1=${ORIG}${TITLE}

# /var/lib/tomcat7
#cd  ~/.config/wbia/tomcat/

# Reset Wildbook database
python -m wbia purge_local_wildbook && python -m wbia install_wildbook
python -m wbia startup_wildbook_server

#python -m wbia test_wildbook_login
wmctrl -xa WILDBOOK_TEST_TERM

#tail -f ~/.config/wbia/tomcat/logs/catalina.out&
sleep 1
xdotool key ctrl+shift+i
xdotool type --clearmodifiers "tail -f ~/.config/wbia/tomcat/logs/catalina.out&"
xdotool key KP_Enter

# Start IA server
sleep 1
xdotool key ctrl+shift+i
xdotool type --clearmodifiers "python -m wbia --web --db PZ_MTEST"
xdotool key KP_Enter


# POSTMAN
# http://127.0.0.1:8080/wbia/ia
# Content-type:application/json; charset=UTF8`
#{
#    "resolver": {
#        "fromIAImageSet": "76817f74-894c-43a4-9a12-47cbc0dc2fc2"
#    }
#}

# http://localhost:8080/wbia/occurrence.jsp?number=76817f74-894c-43a4-9a12-47cbc0dc2fc2


#sudo apt-get install mysql-server-5.6
#sudo apt-get install mysql-common-5.6
#sudo apt-get install mysql-client-5.6

#mysql -u root -p

#create user 'wbiawb'@'localhost' identified by 'somepassword';
#create database wbiawbtestdb;
#grant all privileges on wbiawbtestdb.* to 'wbiawb'@'localhost';


## Test an log in
#mysql -u wbiawb -p wbiawbtestdb
