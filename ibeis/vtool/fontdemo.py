#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Needs freetype-py>=1.0
# The MIT License (MIT)
#
# Copyright (c) 2013 Daniel Bader (http://dbader.org)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
References:
    https://dbader.org/blog/monochrome-font-rendering-with-freetype-and-python
    https://gist.github.com/dbader/5488053#file-fontdemo-py-L244
"""
from __future__ import absolute_import, division, print_function, unicode_literals


class Bitmap(object):
    """
    A 2D bitmap image represented as a list of byte values. Each byte indicates the state
    of a single pixel in the bitmap. A value of 0 indicates that the pixel is `off`
    and any other value indicates that it is `on`.
    """
    def __init__(self, width, height, pixels=None):
        self.width = width
        self.height = height
        self.pixels = pixels or bytearray(width * height)

    def __repr__(self):
        """Return a string representation of the bitmap's pixels."""
        rows = ''
        for y in range(self.height):
            for x in range(self.width):
                rows += '#' if self.pixels[y * self.width + x] else '.'
            rows += '\n'
        return rows

    def bitblt(self, src, x, y):
        """Copy all pixels from `src` into this bitmap"""
        srcpixel = 0
        dstpixel = y * self.width + x
        row_offset = self.width - src.width
        for sy in range(src.height):
            for sx in range(src.width):
                self.pixels[dstpixel] = self.pixels[dstpixel] or src.pixels[srcpixel]
                srcpixel += 1
                dstpixel += 1
            dstpixel += row_offset


class Glyph(object):
    def __init__(self, pixels, width, height, top, advance_width):
        self.bitmap = Bitmap(width, height, pixels)
        self.top = top
        self.descent = max(0, self.height - self.top)
        self.ascent = max(0, max(self.top, self.height) - self.descent)
        self.advance_width = advance_width

    @property
    def width(self):
        return self.bitmap.width

    @property
    def height(self):
        return self.bitmap.height

    @staticmethod
    def from_glyphslot(slot):
        """Construct and return a Glyph object from a FreeType GlyphSlot."""
        pixels = Glyph.unpack_mono_bitmap(slot.bitmap)
        width, height = slot.bitmap.width, slot.bitmap.rows
        top = slot.bitmap_top
        advance_width = slot.advance.x // 64
        return Glyph(pixels, width, height, top, advance_width)

    @staticmethod
    def unpack_mono_bitmap(bitmap):
        """
        Unpack a freetype FT_LOAD_TARGET_MONO glyph bitmap into a bytearray where each
        pixel is represented by a single byte.
        """
        data = bytearray(bitmap.rows * bitmap.width)
        for y in range(bitmap.rows):
            for byte_index in range(bitmap.pitch):
                byte_value = bitmap.buffer[y * bitmap.pitch + byte_index]
                num_bits_done = byte_index * 8
                rowstart = y * bitmap.width + byte_index * 8
                for bit_index in range(min(8, bitmap.width - num_bits_done)):
                    bit = byte_value & (1 << (7 - bit_index))
                    data[rowstart + bit_index] = 1 if bit else 0
        return data


class Font(object):
    def __init__(self, filename, size):
        import freetype
        self.face = freetype.Face(filename)
        self.face.set_pixel_sizes(0, size)

    def glyph_for_character(self, char):
        import freetype
        self.face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
        return Glyph.from_glyphslot(self.face.glyph)

    def render_character(self, char):
        glyph = self.glyph_for_character(char)
        return glyph.bitmap

    def kerning_offset(self, previous_char, char):
        kerning = self.face.get_kerning(previous_char, char)
        return kerning.x // 64

    def text_dimensions(self, text):
        """
        Return (width, height, baseline) of `text` rendered in the current
        font. """
        width = 0
        max_ascent = 0
        max_descent = 0
        previous_char = None
        for char in text:
            glyph = self.glyph_for_character(char)
            max_ascent = max(max_ascent, glyph.ascent)
            max_descent = max(max_descent, glyph.descent)
            kerning_x = self.kerning_offset(previous_char, char)
            width += max(glyph.advance_width + kerning_x, glyph.width + kerning_x)
            previous_char = char
        height = max_ascent + max_descent
        return (width, height, max_descent)

    def render_text(self, text, width=None, height=None, baseline=None):
        """
        Render the given `text` into a Bitmap and return it.  If `width`,
        `height`, and `baseline` are not specified they are computed using the
        `text_dimensions' method.
        """
        if None in (width, height, baseline):
            width, height, baseline = self.text_dimensions(text)
        x = 0
        previous_char = None
        outbuffer = Bitmap(width, height)
        for char in text:
            glyph = self.glyph_for_character(char)
            x += self.kerning_offset(previous_char, char)
            y = height - glyph.ascent - baseline
            outbuffer.bitblt(glyph.bitmap, x, y)
            x += glyph.advance_width
            previous_char = char
        return outbuffer


def font_demo():
    r"""
    CommandLine:
        python -m vtool_ibeis.fontdemo font_demo --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.fontdemo import *  # NOQA
        >>> result = font_demo()
        >>> import utool as ut
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> ut.show_if_requested()
    """
    filename = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
    fnt = Font(filename, 12)
    # Single characters
    ch = fnt.render_character('e')
    print(repr(ch))
    # Multiple characters
    txt = fnt.render_text('hello')
    print(repr(txt))
    # Kerning
    print(repr(fnt.render_text('A1321')))
    # Choosing the baseline correctly
    print(repr(fnt.render_text('hello world')))


def get_text_test_img(text):
    r"""
    Args:
        text (str):

    CommandLine:
        python -m vtool_ibeis.fontdemo get_text_test_img --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.fontdemo import *  # NOQA
        >>> import utool as ut
        >>> text = 'A012'
        >>> text_img = get_text_test_img(text)
        >>> result = ('text_img = %s' % (ub.repr2(text_img),))
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.imshow(text_img)
        >>> ut.show_if_requested()
    """
    filename = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
    fnt = Font(filename, 24)
    buf = fnt.render_text(text)
    import numpy as np
    img = np.array(buf.pixels)
    text_img = img.reshape(buf.height, buf.width) * 255
    return text_img

if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool_ibeis.fontdemo
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
