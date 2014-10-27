from __future__ import absolute_import, division, print_function
# Python
from six.moves import zip
from os.path import exists, join
# UTool
import utool
# VTool
import vtool.chip as ctool
import vtool.image as gtool
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[preproc_detectimg]', DEBUG=False)


def gen_detectimg_and_write(tup):
    """ worker function for parallel generator """
    gid, gfpath, new_gfpath, new_size = tup
    #print('[preproc] writing detectimg: %r' % new_gfpath)
    img = gtool.imread(gfpath)
    new_img = gtool.resize(img, new_size)
    gtool.imwrite(new_gfpath, new_img)
    return gid, new_gfpath


def gen_detectimg_async(gid_list, gfpath_list, new_gfpath_list,
                        newsize_list, nImgs=None):
    """ Resizes images and yeilds results asynchronously  """
    # Compute and write detectimg in asychronous process
    if nImgs is None:
        nImgs = len(gid_list)
    arg_iter = zip(gid_list, gfpath_list, new_gfpath_list, newsize_list)
    arg_list = list(arg_iter)
    return utool.util_parallel.generate(gen_detectimg_and_write, arg_list)


def get_image_detectimg_fpath_list(ibs, gid_list):
    """ Returns detectimg path list """
    utool.assert_all_not_None(gid_list, 'gid_list')
    gext_list    = ibs.get_image_exts(gid_list)
    guuid_list   = ibs.get_image_uuids(gid_list)
    cachedir = ibs.get_detectimg_cachedir()
    new_gfpath_list = [join(cachedir, 'reszd_' + str(guuid) + ext)
                       for (guuid, ext) in zip(guuid_list, gext_list)]
    return new_gfpath_list


def compute_and_write_detectimg(ibs, gid_list):
    utool.ensuredir(ibs.get_detectimg_cachedir())
    # Get img dest information (output path)
    new_gfpath_list = get_image_detectimg_fpath_list(ibs, gid_list)
    # Get img configuration information
    sqrt_area   = 800  # TODO: Put this in a config
    target_area = sqrt_area ** 2
    # Get img source information (image, annotation_bbox, theta)
    gfpath_list  = ibs.get_image_paths(gid_list)
    gsize_list   = ibs.get_image_sizes(gid_list)
    newsize_list = ctool.get_scaled_sizes_with_area(target_area, gsize_list)
    # Define "Asynchronous" generator
    print('[preproc_detectimg] Computing %d imgs asynchronously' % (len(gfpath_list)))
    detectimg_async_iter = gen_detectimg_async(gid_list, gfpath_list,
                                               new_gfpath_list, newsize_list)
    for gid, new_gfpath in detectimg_async_iter:
        # print('Wrote detectimg: %r' % new_gfpath)
        pass
    print('[preproc_detectimg] Done computing detectimgs')


def compute_and_write_detectimg_lazy(ibs, gid_list):
    """
    Will write a img if it does not exist on disk, regardless of if it exists
    in the SQL database
    """
    print('[preproc] compute_and_write_detectimg_lazy')
    # Mark which aid's need their detectimg computed
    new_gfpath_list = get_image_detectimg_fpath_list(ibs, gid_list)
    exists_flags = [exists(gfpath) for gfpath in new_gfpath_list]
    invalid_gids = utool.get_dirty_items(gid_list, exists_flags)
    print('[preproc_detectimg] %d / %d detectimgs need to be computed' %
          (len(invalid_gids), len(gid_list)))
    compute_and_write_detectimg(ibs, invalid_gids)
    return new_gfpath_list
