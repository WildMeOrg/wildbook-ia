# -*- coding: utf-8 -*-
"""
Security component

Provides access to the werkzeug security python utils to generate random strings with extra helpers
"""

from werkzeug import security


# A random string with length determined by parameter
def generate_random(length):
    return security.gen_salt(length)


def generate_random_64():
    return generate_random(64)


def generate_random_128():
    return generate_random(128)
