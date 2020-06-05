# DISABLE THESE ERRORS
#'E127', # continuation line over-indented for visual indent
#'E201', # whitespace after '('
#'E202', # whitespace before ']'
#'E203', # whitespace before ', '
#'E221', # multiple spaces before operator
#'E222', # multiple spaces after operator
#'E241', # multiple spaces after ,
#'E265', # block comment should start with "# "
#'E271', # multiple spaces after keyword
#'E272', # multiple spaces before keyword
#'E301', # expected 1 blank line, found 0
#'E501', # > 79

export FLAKE8_IGNORE="--ignore=E127,E201,E202,E203,E221,E222,E241,E265,E271,E272,E301,E501"

flake8 $FLAKE8_IGNORE ~/code/wbia
