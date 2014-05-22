gg-recover()
{
    git checkout $(git rev-list -n 1 HEAD -- "$1")^ -- "$1"
}

gg-stats()
{
    python ~/local/util_git.py 'git status'
}

gg-pull()
{
    python ~/local/util_git.py 'git pull'
}

gg-push()
{
    python ~/local/util_git.py 'git push'
}

alias ggs=gg-stats
