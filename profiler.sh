#!/bin/sh
# Globals
export TIMESTAMP=$(date -d "today" +"%Y-%m-%d_%H-%M-%S")
export RAW_SUFFIX=raw.prof
export CLEAN_SUFFIX=clean.prof

# Input
export pyscript=$1
# Output Stage 1
export line_profile_output=$pyscript.lprof
# Output Stage 2
export raw_profile=raw_profile.$pyscript.$TIMESTAMP.$RAW_SUFFIX
export clean_profile=clean_profile.$pyscript.$TIMESTAMP.$CLEAN_SUFFIX

remove_profiles()
{
    echo "Removing profiles"
    #rm *.profile.txt
    rm *raw.prof
    rm *clean.prof
    rm *.lprof
    echo "Finshed removing profiles"
}

echo "Input: $@"


if [ "$pyscript" = "clean" ]; then
    remove_profiles
    exit
fi 

echo "pyscript: $pyscript"
echo "line_profile_output: $line_profile_output"
echo "Profiling $pyscript"

export SYSNAME="$(expr substr $(uname -s) 1 10)"
# Choose one
export PROFILE_TYPE="kernprof"  # plop or kernprof
#export PROFILE_TYPE="plop"  # plop or kernprof


if [ $PROFILE_TYPE = "plop" ]; then
    python -m plop.collector $@
    echo "http://localhost:8888"
    python -m plop.viewer --datadir=profiles
elif [ $PROFILE_TYPE = "kernprof" ]; then
    # Line profile the python code w/ command line args
    if [ "$SYSNAME" = "MINGW32_NT" ]; then
        export MINGW_PYEXE=$(python -c "import sys; print(sys.executable)")
        export MINGW_PYDIR=$(python -c "import sys, os; print(os.path.dirname(sys.executable))")
        $MINGW_PYEXE $MINGW_PYDIR/Scripts/kernprof.py -l $@
    else
        kernprof.py -l $@
    fi
    # Dump the line profile output to a text file
    python -m line_profiler $line_profile_output >> $raw_profile
    # Clean the line profile output
    python _scripts/profiler_cleaner.py $raw_profile $clean_profile
    # Print the cleaned output
    cat $clean_profile
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
