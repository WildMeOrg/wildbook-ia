from __future__ import absolute_import, division, print_function
import numpy as np
import decimal
from .util_type import is_float, is_int
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[num]')


def order_of_magnitude_ceil(num):
    nDigits = np.ceil(np.log10(num))
    scalefactor = 10 ** (nDigits - 1)
    return np.ceil(num / scalefactor) * scalefactor


def format_(num, n=8):
    """
        makes numbers pretty e.g.
        nums = [9001, 9.053]
        print([format_(num) for num in nums])
    """
    if num is None:
        return 'None'
    if is_float(num):
        ret = ('%.' + str(n) + 'E') % num
        exp_pos  = ret.find('E')
        exp_part = ret[(exp_pos + 1):]
        exp_part = exp_part.replace('+', '')
        if exp_part.find('-') == 0:
            exp_part = '-' + exp_part[1:].strip('0')
        exp_part = exp_part.strip('0')
        if len(exp_part) > 0:
            exp_part = 'E' + exp_part
        flt_part = ret[:exp_pos].strip('0').strip('.')
        ret = flt_part + exp_part
        return ret
    return '%d' % num


def float_to_decimal(f):
    # http://docs.python.org/library/decimal.html#decimal-faq
    "Convert a floating point number to a Decimal with no loss of information"
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result


#http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python
def sigfig_str(number, sigfig):
    # http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python/2663623#2663623
    assert(sigfig > 0)
    try:
        d = decimal.Decimal(number)
    except TypeError:
        d = float_to_decimal(float(number))
    sign, digits, exponent = d.as_tuple()
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    shift = d.adjusted()
    result = int(''.join(map(str, digits[:sigfig])))
    # Round the result
    if len(digits) > sigfig and digits[sigfig] >= 5:
        result += 1
    result = list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift += len(result) - sigfig
    # reset len of result to sigfig
    result = result[:sigfig]
    if shift >= sigfig - 1:
        # Tack more zeros on the end
        result += ['0'] * (shift - sigfig + 1)
    elif 0 <= shift:
        # Place the decimal point in between digits
        result.insert(shift + 1, '.')
    else:
        # Tack zeros on the front
        assert(shift < 0)
        result = ['0.'] + ['0'] * (-shift - 1) + result
    if sign:
        result.insert(0, '-')
    return ''.join(result)


def num2_sigfig(num):
    return int(np.ceil(np.log10(num)))


def num_fmt(num, max_digits=1):
    if is_float(num):
        return ('%.' + str(max_digits) + 'f') % num
    elif is_int(num):
        return int_comma_str(num)
    else:
        return '%r'


def int_comma_str(num):
    int_str = ''
    reversed_digits = decimal.Decimal(num).as_tuple()[1][::-1]
    for i, digit in enumerate(reversed_digits):
        if (i) % 3 == 0 and i != 0:
            int_str += ','
        int_str += str(digit)
    return int_str[::-1]


def fewest_digits_float_str(num, n=8):
    int_part = int(num)
    dec_part = num - int_part
    x = decimal.Decimal(dec_part, decimal.Context(prec=8))
    decimal_list = x.as_tuple()[1]
    nonzero_pos = 0
    for i in range(0, min(len(decimal_list), n)):
        if decimal_list[i] != 0:
            nonzero_pos = i
    sig_dec = int(dec_part * 10 ** (nonzero_pos + 1))
    float_str = int_comma_str(int_part) + '.' + str(sig_dec)
    return float_str
    #x.as_tuple()[n]


def commas(num, n=8):
    if is_float(num):
        #ret = sigfig_str(num, n=2)
        ret = '%.3f' % num
        return ret
        #return fewest_digits_float_str(num, n)
    return '%d' % num
    #return int_comma_str(num)
