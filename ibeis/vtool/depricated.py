

class ThumbnailCacheContext(object):
    """ Lazy computation of of images as thumbnails.

    DEPRICATED

    Just pass a list of uuids corresponding to the images. Then compute images
    flagged as dirty and give them back to the context.  thumbs_list will be
    populated on contex exit
    """
    def __init__(self, uuid_list, asrgb=True, thumb_size=64, thumb_dpath=None, appname='vtool'):
        if thumb_dpath is None:
            # Get default thumb path
            thumb_dpath = ut.get_app_resource_dir(appname, 'thumbs')
        ut.ensuredir(thumb_dpath)
        self.thumb_gpaths = [join(thumb_dpath, str(uuid) + 'thumb.png') for uuid in uuid_list]
        self.asrgb = asrgb
        self.thumb_size = thumb_size
        self.thumb_list = None
        self.dirty_list = None
        self.dirty_gpaths = None

    def __enter__(self):
        # These items need to be computed
        self.dirty_list = [not exists(gpath) for gpath in self.thumb_gpaths]
        self.dirty_gpaths = ut.filter_items(self.thumb_gpaths, self.dirty_list)
        #print('[gtool.thumb] len(dirty_gpaths): %r' % len(self.dirty_gpaths))
        self.needs_compute = len(self.dirty_gpaths) > 0
        return self

    def save_dirty_thumbs_from_images(self, img_list):
        """ Pass in any images marked by the context as dirty here """
        # Remove any non images
        isvalid_list = [img is not None for img in img_list]
        valid_images  = ut.filter_items(img_list, isvalid_list)
        valid_fpath = ut.filter_items(self.thumb_gpaths, isvalid_list)
        # Resize to thumbnails
        max_dsize = (self.thumb_size, self.thumb_size)
        valid_thumbs = [resize_thumb(img, max_dsize) for img in valid_images]
        # Write thumbs to disk
        for gpath, thumb in zip(valid_fpath, valid_thumbs):
            imwrite(gpath, thumb)

    def filter_dirty_items(self, list_):
        """ Returns only items marked by the context as dirty """
        return ut.filter_items(list_, self.dirty_list)

    def __exit__(self, type_, value, trace):
        if trace is not None:
            print('[gtool.thumb] Error while in thumbnail context')
            print('[gtool.thumb] Error in context manager!: ' + str(value))
            return False  # return a falsey value on error
        # Try to read thumbnails on disk
        self.thumb_list = [_trimread(gpath) for gpath in self.thumb_gpaths]
        if self.asrgb:
            self.thumb_list = [None if thumb is None else cvt_BGR2RGB(thumb)
                               for thumb in self.thumb_list]
