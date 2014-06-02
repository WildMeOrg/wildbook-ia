#!/bin/sh
# Globals
echo "==========="
echo "PROFILER.sh"
export TIMESTAMP=$(date -d "today" +"%Y-%m-%d_%H-%M-%S")
export RAW_SUFFIX=raw.prof
export CLEAN_SUFFIX=clean.prof


remove_profiles()
{
    echo "Removing profiles"
    #rm *.profile.txt
    rm *raw.prof
    rm *clean.prof
    rm *.lprof
    echo "Finshed removing profiles"
}

# Input (get only the filename)

export pyscript=$(python -c "import os; print(os.path.split(r'$1')[1])")
#export pyscript=$(echo $1 | sed -e 's/.*[\/\\]//g')

if [ "$pyscript" = "clean" ]; then
    remove_profiles
    exit
fi 

#echo $pyscript
#set -e
#/bin/command-that-fails

export SYSNAME="$(expr substr $(uname -s) 1 10)"

# Choose one # plop or lineprof, or runsnake
#export PROFILE_TYPE="runsnake"  
export PROFILE_TYPE="lineprof"

# Platform independent kernprof
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export MINGW_PYEXE=$(python -c "import sys; print(sys.executable)")
    export MINGW_PYDIR=$(python -c "import sys, os; print(os.path.dirname(sys.executable))")
    export MINGW_PYSCRIPTS=$MINGW_PYDIR/Scripts
    export KERNPROF_PY="$MINGW_PYEXE $MINGW_PYDIR/Scripts/kernprof.py"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export PORT_PYEXE=$(python -c "import sys; print(sys.executable)")
    export PORT_PYSCRIPTS="/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin"
    echo $PORT_PYEXE
    echo $PORT_PYDIR
    export KERNPROF_PY="$PORT_PYEXE $PORT_PYDIR/Scripts/kernprof.py"
else
    export KERNPROF_PY="kernprof.py"
    export RUNSNAKE_PY="runsnake"
fi

echo "pyscript: $pyscript"

#
#
# PLOP PROFILER
if [ $PROFILE_TYPE = "plop" ]; then
    python -m plop.collector $@
    echo "http://localhost:8888"
    python -m plop.viewer --datadir=profiles
#
#
# LINE PROFILER
elif [ $PROFILE_TYPE = "lineprof" ]; then
    # Output Stage 1
    echo "line_profile_output: $line_profile_output"
    export line_profile_output=$pyscript.lprof
    # Output Stage 2
    export raw_profile=raw_profile.$pyscript.$TIMESTAMP.$RAW_SUFFIX
    export clean_profile=clean_profile.$pyscript.$TIMESTAMP.$CLEAN_SUFFIX
    # Line profile the python code w/ command line args
    $KERNPROF_PY -l $@
    # Dump the line profile output to a text file
    python -m line_profiler $line_profile_output >> $raw_profile
    # Clean the line profile output
    python _scripts/profiler_cleaner.py $raw_profile $clean_profile
    # Print the cleaned output
    cat $clean_profile
#
#
# PLOP PROFILER
elif [ $PROFILE_TYPE = "runsnake" ]; then
    # Line profile the python code w/ command line args
    export runsnake_profile=RunSnakeContext.$pyscript.$TIMESTAMP.$RAW_SUFFIX
    python -m cProfile -o $runsnake_profile $@
    # Dump the line profile output to a text file
    $RUNSNAKE_PY $runsnake_profile
fi

# In windows I had to set the default .sh extension to run with
# C:\MinGW\msys\1.0\bin\sh.exe
#assoc .sh=sh_auto_file
#ftype sh_auto_file="C:\MinGW\msys\1.0\bin\sh.exe" %1 %*
#
#set PATHEXT=.sh;%PATHEXT% 
#assoc .sh
#assoc .sh=bashscript
#ftype bashscript="C:\MinGW\msys\1.0\bin\sh.exe" %1 %*
#https://stackoverflow.com/questions/105075/how-can-i-associate-sh-files-with-cygwin
