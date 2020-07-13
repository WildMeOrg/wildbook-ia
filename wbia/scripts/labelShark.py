# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.scripts import classify_shark
import numpy as np
import vtool as vt
import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[sharkspotter]')

modelStateLocation = (
    'https://wildbookiarepository.azureedge.net/models/classifier.lenet.whale_shark.pkl'
)


def classifyShark(ibs, gid_list):

    suffix = 'lenet'
    batch_size = 32
    model_name = 'injur-shark-' + suffix
    model = classify_shark.WhaleSharkInjuryModel(
        name=model_name, output_dims=2, data_shape=(224, 224, 3), batch_size=batch_size,
    )
    model.init_arch()
    filep = ut.grab_file_url(modelStateLocation)
    model.load_model_state(fpath=filep)
    model.rrr()

    config = {
        'algo': 'yolo',
        'sensitivity': 0.2,
        'config_filepath': 'default',
    }
    depc = ibs.depc_image

    images = ibs.images(gid_list)
    images = images.compress([ext_ not in ['.gif'] for ext_ in images.exts])

    gid_list = images.gids
    # uuid_gid_list = [str(item) for item in images.uuids]
    results_list = depc.get_property(
        'localizations', gid_list, None, config=config
    )  # NOQA

    results_list2 = []
    multi_gids = []
    failed_gids = []

    for gid, res in zip(gid_list, results_list):
        score, bbox_list, theta_list, conf_list, class_list = res
        if len(bbox_list) == 0:
            failed_gids.append(gid)
        elif len(bbox_list) == 1:
            results_list2.append((gid, bbox_list, theta_list))
        elif len(bbox_list) > 1:
            # Take only a single annotation per bounding box.
            multi_gids.append(gid)
            idx = conf_list.argmax()
            res2 = (gid, bbox_list[idx : idx + 1], theta_list[idx : idx + 1])
            results_list2.append(res2)

    # Reorder empty_info to be aligned with results
    localized_imgs = ibs.images(ut.take_column(results_list2, 0))

    # Override old bboxes
    bboxes = np.array(ut.take_column(results_list2, 1))[:, 0, :]
    thetas = np.array(ut.take_column(results_list2, 2))[:, 0]

    species = ['whale_shark'] * len(localized_imgs)
    aid_list = ibs.add_annots(
        localized_imgs.gids, bbox_list=bboxes, theta_list=thetas, species_list=species
    )

    config = {'dim_size': (224, 224), 'resize_dim': 'wh'}
    chip_gen = ibs.depc_annot.get('chips', aid_list, 'img', eager=False, config=config)
    data_shape = config['dim_size'] + (3,)
    iter_ = iter(ut.ProgIter(chip_gen, nTotal=len(aid_list), lbl='load chip'))
    shape = (len(aid_list),) + data_shape
    data = vt.fromiter_nd(iter_, shape=shape, dtype=np.uint8)  # NOQA
    results = model._predict(data)
    predictions = results['predictions']
    classes = np.array(['healthy', 'injured'])
    prediction_class = classes[np.array(predictions)]
    return {
        'predictions': prediction_class.tolist(),
        'confidences': results['confidences'].tolist(),
    }
