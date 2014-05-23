import utool
from utool._internal.meta_util_git import repo_list, set_userid
utool.repo_list = repo_list

print('__IBEIS_SUPER_SETUP__')


def userid_prompt():
    return 'Erotemic'

set_userid(userid_prompt())
CODE_DIR = utool.unixpath('~/code')

# Non local project repos
(IBEIS_REPOS_URLS,
 IBEIS_REPOS) = utool.repo_list([
    'https://github.com/Erotemic/opencv.git',
    'https://github.com/Erotemic/flann.git',
    'https://github.com/Erotemic/utool.git',
    'https://github.com/Erotemic/guitool.git',
    'https://github.com/Erotemic/plottool.git',
    'https://github.com/Erotemic/vtool.git',
    'https://github.com/Erotemic/hesaff.git',
    'https://github.com/Erotemic/ibeis.git',
    'https://github.com/bluemellophone/pyrf.git',
    'https://github.com/bluemellophone/detecttools.git',
], CODE_DIR)

## Option A
#for repo in IBEIS_REPOS:
#    if not utool.checkpath(repo, verbose=True):
#        utool.cmd('git pull ' + repo)

# Option B
utool.util_git.PROJECT_REPOS = IBEIS_REPOS

for repo, url in utool.izip(IBEIS_REPOS, IBEIS_REPOS):
    if not utool.exists(repo):
        print('repo does not exist')
        #dirname(repo)

if utool.get_flag('--status'):
    utool.gg_command('git status')
    utool.sys.exit(0)

if utool.get_flag('--pull'):
    utool.gg_command('git pull')
