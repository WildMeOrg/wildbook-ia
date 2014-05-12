# NonPip modules


# PyQt4
sudo apt-get install python-qt4



# Apt-Get

sudo apt-get update
sudo apt-get upgrade


# Wget


# Wget SQLite3 v3.8.4.1
cd ~/tmp
wget https://sqlite.org/2014/sqlite-autoconf-3080401.tar.gz
7z x sqlite-autoconf-3080401.tar.gz
7z x sqlite-autoconf-3080401.tar
cd sqlite-autoconf-3080401
chmod +x configure
./configure
make
sudo make install
# Pip modules


    fixme-python-sqlite3()
    {
    # Need SQLite >= 3.7.11
    #https://www.sqlite.org/download.html
    sudo pip install db-sqlite3

    sudo pip install --upgrade --force-reinstall --install-option="build_static" pysqlite

    echo "SQLite3 version should be >= 3.7.11"
    python -c "import sqlite3; print(sqlite3.sqlite_version)"


    python -c "from pysqlite2 import dbapi2 as sqlite3; print(sqlite3.sqlite_version)"

    # Try and build everything from source
    sudo pip uninstall pysqlite

    cd ~/tmp
    wget https://pypi.python.org/packages/source/p/pysqlite/pysqlite-2.6.3.tar.gz#md5=7ff1cedee74646b50117acff87aa1cfa

    tar xzvf pysqlite-2.6.3.tar.gz
    cd pysqlite3

    ed setup.cfg <<EOF
    /SQLITE_OMIT_LOAD_EXTENSION/s/define=/#define=/
    w
    q
    EOF

    wget https://www.sqlite.org/2014/sqlite-amalgamation-3080401.zip
    7z x sqlite-amalgamation-3080401.zip
    mv sqlite-amalgamation-3080401/*.[hc] .

    sudo python setup.py build_static
    sudo python setup.py install

    python <<EOF
    from pysqlite2 import test
    test.test()
    EOF

    python <<EOF
    from pysqlite2 import dbapi2 as sqlite3
    con = sqlite3.connect(":memory:")
    res = con.execute("create virtual table recipe using fts3(name, ingredients)")
    print(res)
    EOF
    }

# SIP
# need newer version to use setdestroyonexit
#sudo pip install --upgrade --allow-external sip --allow-unverified sip sip}
