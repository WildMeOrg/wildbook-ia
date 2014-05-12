from __future__ import absolute_import, division, print_function
import hashlib
import uuid
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[hash]')

# default length of hash codes
HASH_LEN = 16

# A large base-54 alphabet (all chars are valid for filenames but not # pretty)
ALPHABET_54 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z', ';', '=', '@', '[',
               ']', '^', '_', '`', '{', '}', '~', '!', '#', '$',
               '%', '&', '+', ',']


# A large base-42 alphabet (prettier subset of base 54)
ALPHABET_42 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z', '@', '!', '$', '%',
               '&', '+']

ALPHABET = ALPHABET_42
BIGBASE = len(ALPHABET)


def hashstr_arr(arr, lbl='arr', **kwargs):
    if isinstance(arr, list):
        arr = tuple(arr)
    if isinstance(arr, tuple):
        arr_shape = '(' + str(len(arr)) + ')'
    else:
        arr_shape = str(arr.shape).replace(' ', '')
    arr_hash = hashstr(arr, **kwargs)
    arr_uid = ''.join((lbl, '(', arr_shape, arr_hash, ')'))
    return arr_uid


def hashstr(data, hashlen=HASH_LEN):
    if isinstance(data, tuple):
        data = repr(data)
    # Get a 128 character hex string
    hashstr = hashlib.sha512(data).hexdigest()
    # Convert to base 54
    hashstr2 = convert_hexstr_to_bigbase(hashstr)
    # Truncate
    hashstr = hashstr2[:hashlen]
    return hashstr


"""
def valid_filename_ascii_chars():
    # Find invalid chars
    ntfs_inval = '< > : " / \ | ? *'.split(' ')
    other_inval = [' ', '\'', '.']
    #case_inval = map(chr, xrange(97, 123))
    case_inval = map(chr, xrange(65, 91))
    invalid_chars = set(ntfs_inval + other_inval + case_inval)
    # Find valid chars
    valid_chars = []
    for index in xrange(32, 127):
        char = chr(index)
        if not char in invalid_chars:
            print index, chr(index)
            valid_chars.append(chr(index))
    return valid_chars
valid_filename_ascii_chars()
"""


def convert_hexstr_to_bigbase(hexstr):
    x = int(hexstr, 16)  # first convert to base 16
    if x == 0:
        return '0'
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(ALPHABET[x % BIGBASE])
        x //= BIGBASE
    if sign < 0:
        digits.append('-')
        digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str


def hashstr_md5(data):
    hashstr = hashlib.md5(data).hexdigest()
    #bin(int(my_hexdata, scale))
    return hashstr


def hashstr_sha1(data, base10=False):
    hashstr = hashlib.sha1(data).hexdigest()
    if base10:
        hashstr = int("0x" + hashstr, 0)

    return hashstr


def hash_to_uuid(bytes_):
    # Hash the bytes
    bytes_sha1 = hashlib.sha1(bytes_)
    # Digest them into a hash
    #hashstr_40 = img_bytes_sha1.hexdigest()
    #hashstr_32 = hashstr_40[0:32]
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)
    return uuid_


def image_uuid(pil_img):
    """ image global unique id """
    # Get the bytes of the image
    img_bytes_ = pil_img.tobytes()
    uuid_ = hash_to_uuid(img_bytes_)
    return uuid_


def augment_uuid(uuid_, *hashables):
    uuidhex_bytes   = uuid_.get_bytes()
    hashable_str    = ''.join(map(repr, hashables))
    augmented_str   = uuidhex_bytes + hashable_str
    augmented_uuid_ = hash_to_uuid(augmented_str)
    return augmented_uuid_


def __test_augment__():
    uuid_ = uuid.uuid1()

    uuidhex_bytes = uuid_.get_bytes()
    hashable_str1 = '[0, 0, 100, 100]'
    hashable_str2 = ''
    augmented_str1 = uuidhex_bytes + hashable_str1
    augmented_str2 = uuidhex_bytes + hashable_str2

    augmented_uuid1_ = hash_to_uuid(augmented_str1)
    augmented_uuid2_ = hash_to_uuid(augmented_str2)

    print('augmented_str1 =%r' % augmented_str1)
    print('augmented_str2 =%r' % augmented_str2)

    print('           uuid_=%r' % (uuid_,))
    print('augmented_uuid1_=%r' % (augmented_uuid1_,))
    print('augmented_uuid2_=%r' % (augmented_uuid2_,))
    print('hash2uuid(uuid_)=%r' % (hash_to_uuid(uuid_),))


def get_zero_uuid():
    return uuid.UUID('00000000-0000-0000-0000-000000000000')

# Cleanup namespace
del ALPHABET_42
del ALPHABET_54
