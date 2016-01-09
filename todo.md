# IBEIS TODO
 
Standardize Config
- Write QT / Web Interface for modifying configs





* TO DEPRICATE
```
params.py in favor of ut.get_argflag and ut.get_argval
```



TO MOVE 
```

cd ~/code/ibeis


def depricate_module(pattern):
    fpaths = ut.glob('.', pattern)
    broken_dpath = join(repo_dpath, '_broken')
    modname_list = ut.lmap(ut.get_modname_from_modpath, fpaths)
    # check_usage(modname_list)
    ut.gg_move(fpaths, broken_dpath)


depricate_module ibeis/algo/hots/automatch_*.py
depricate_module ibeis/algo/hots/user_dialogs.py
depricate_module ibeis/algo/hots/special_query.py
depricate_module ibeis/algo/hots/qt_inc_automatch.py
depricate_module ibeis/algo/hots/_grave*.py

move_module ibeis/algo/hots/user_dialogs.py
```

TO CREATE:
```
init_module hots/ibeis_workflow.py
```
