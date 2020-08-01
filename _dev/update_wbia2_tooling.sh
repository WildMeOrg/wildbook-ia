__heredoc__='''
Script to use the pypi modules
'''


sedr_python(){
    SEARCH=$1
    REPLACE=$2
    LIVE_RUN=$3

    echo "
    === sedr python ===
    SEARCH = '$SEARCH'
    REPLACE = '$REPLACE'
    LIVE_RUN = '$LIVE_RUN'
    "

    if [[ "$LIVE_RUN" == "True" ]]; then
        find . -type f -iname '*.py' -exec sed -i "s/${SEARCH}/${REPLACE}/g" {} +
    else
        find . -type f -iname '*.py' -exec sed "s/${SEARCH}/${REPLACE}/g" {} + | grep "${REPLACE}"
    fi
}

LIVE_RUN=False
#LIVE_RUN=True


TOOL_NAME="vtool"
sedr_python "\\b${TOOL_NAME}\\b" "${TOOL_NAME}_wbia" $LIVE_RUN
TOOL_NAME="dtool"
sedr_python "\\b${TOOL_NAME}\\b" "${TOOL_NAME}_wbia" $LIVE_RUN
TOOL_NAME="plottool"
sedr_python "\\b${TOOL_NAME}\\b" "${TOOL_NAME}_wbia" $LIVE_RUN
TOOL_NAME="guitool"
sedr_python "\\b${TOOL_NAME}\\b" "${TOOL_NAME}_wbia" $LIVE_RUN

TOOL_NAME="pyflann"
sedr_python "import pyflann" "from vtool._pyflann_backend import pyflann as pyflann" $LIVE_RUN
sedr_python "from vtool._pyflann_backend" "from vtool_ibeis._pyflann_backend" $LIVE_RUN


sedr_python "# GUI_DOCTEST" "# xdoctest: +REQUIRES(--gui)" True
sedr_python "# WEB_DOCTEST" "# xdoctest: +REQUIRES(--web-tests)" True
