import ibeis
import ibeis_cnn  # NOQA
from ibeis_cnn.ingest_ibeis import get_background_training_patches2


if __name__ == '__main__':
    dbname_list = [
        'ELPH_Master',
        'GIR_Master',
        'GZ_Master',
        'NNP_MasterGIRM',
        'PZ_Master1',
    ]

    global_positives, global_negatives, global_labels = 0, 0, 0
    for dbname in dbname_list:
        print(dbname)
        ibs = ibeis.opendb('/Datasets/BACKGROUND/%s/' % (dbname, ))
        p, n, l = get_background_training_patches2(ibs)

        global_positives += p
        global_negatives += n
        global_labels += l

    args = (global_positives, global_negatives, global_labels, )
    print('FINAL SPLIT: [ %r / %r = %r]' % args)
