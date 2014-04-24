DEVCMD_FUNCTIONS = []


def devcmd(*args):
    """
    Registers a function as a developer command
    """
    global DEVCMD_FUNCTIONS
    if len(args) == 1 and not isinstance(args[0], str):
        func = args[0]
        DEVCMD_FUNCTIONS.append(((func.func_name,), func))
        return func
    else:
        def wrapper(func):
            func_aliases = [func.func_name]
            func_aliases.extend(args)
            DEVCMD_FUNCTIONS.append((tuple(func_aliases), func))
            return func
        return wrapper
