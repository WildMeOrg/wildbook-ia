"""
This script handles the skimage exif problem.
"""

from PIL import Image
import numpy as np
import cStringIO as StringIO

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}


def open_oriented_image(im_path):
    im = Image.open(im_path)
    if hasattr(im, '_getexif'):
        exif = im._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            im = apply_orientation(im, orientation)
    img = np.asarray(im).astype(np.float32) / 255.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def apply_orientation(im, orientation):
    if orientation in ORIENTATIONS:
        for method in ORIENTATIONS[orientation]:
            im = im.transpose(method)
    return im


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    # image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data
