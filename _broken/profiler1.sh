#!/bin/bash
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

#echo $pyscript
#set -e
#/bin/command-that-fails


# Choose one # plop or lineprof, or runsnake
#export PROFILE_TYPE="runsnake"  
export PROFILE_TYPE="lineprof"

# Platform independent kernprof
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    # MINGW
    export MINGW_PYDIR=$(python -c "import sys, os; print(os.path.dirname(sys.executable))")
    export PYEXE=$(python -c "import sys; print(sys.executable)")
    export PYSCRIPTS=$MINGW_PYDIR/Scripts
    export KERNPROF_PY="$PYEXE $PYSCRIPTS/kernprof.py"
elif [ "$OSTYPE" = "darwin"* ]; then
    # MACPORTS
    export PYEXE=$(python -c "import sys; print(sys.executable)")
    export PYSCRIPTS="/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin"
    echo "Python EXE: $PYEXE"
    echo "Python Scripts $PYSCRIPTS"
    export KERNPROF_PY="$PYEXE $PYSCRIPTS/kernprof.py"
else
    # UBUNTU
    #export KERNPROF_PY="kernprof.py"
    export KERNPROF_PY="kernprof"

    export RUNSNAKE_PY="runsnake"
    export PYEXE=$(python -c "import sys; print(sys.executable)")
fi

# Input (get only the filename)

export pyscript=$($PYEXE -c "import os; print(os.path.split(r'$1')[1])")
#export pyscript=$(echo $1 | sed -e 's/.*[\/\\]//g')

if [ "$pyscript" = "clean" ]; then
    remove_profiles
    exit
fi 

echo "pyscript: $pyscript"

#
#
# PLOP PROFILER
if [ $PROFILE_TYPE = "plop" ]; then
    $PYEXE -m plop.collector $@
    echo "http://localhost:8888"
    $PYEXE -m plop.viewer --datadir=profiles
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
    $PYEXE -m line_profiler $line_profile_output >> $raw_profile
    # Clean the line profile output
    $PYEXE _scripts/profiler_cleaner.py $raw_profile $clean_profile
    # Print the cleaned output
    cat $clean_profile
#
#
# PLOP PROFILER
elif [ $PROFILE_TYPE = "runsnake" ]; then
    # Line profile the python code w/ command line args
    export runsnake_profile=RunSnakeContext.$pyscript.$TIMESTAMP.$RAW_SUFFIX
    $PYEXE -m cProfile -o $runsnake_profile $@
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
