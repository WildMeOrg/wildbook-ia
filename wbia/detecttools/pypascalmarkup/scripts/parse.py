# -*- coding: utf-8 -*-
import os
import random
from collections import defaultdict
import xml.etree.ElementTree


# want to map a the filename -> all the animals present in the image
def parse_annotations(dir):
    all_files = [
        f
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and f.lower().endswith('.xml')
    ]
    filenames = defaultdict(list)
    for f in all_files:
        target_file = os.path.join(dir, f)
        # check that the annotation's xml file exists
        if os.path.isfile(target_file):
            print('parsing %s' % target_file)
            with open(target_file, 'r') as xml_file:
                # get the raw xml file from the annotation file
                raw_xml = xml_file.read().replace('\n', '')
                # read it into an Element
                data_xml = xml.etree.ElementTree.XML(raw_xml)
                # get all instances of filename, there should only be one!
                filename_xml = [f for f in data_xml.findall('filename')]
                if len(filename_xml) > 1:
                    print('problem with %s, more than one filename!' % target_file)
                fname = filename_xml[0]
                # get all bounding boxes in this annotation
                for obj in data_xml.findall('object'):
                    # get the animals present in this image, don't want the file extension
                    for classname in obj.findall('name'):
                        filenames[fname.text[0:-4]].append(classname.text)
        else:
            print('could not find %s, ignoring' % target_file)

    # for k in filenames:
    #    print k, filenames[k]

    return filenames


if __name__ == '__main__':
    # the ratio of data to be set aside for training
    training_ratio = 0.8
    # class that will be marked as positive training examples
    classname1 = 'zebra_grevys'
    classname2 = 'zebra_plains'
    # directory that contains the xml annotations
    xml_dir = '/media/IBEIS/data/Annotations'
    annotations = parse_annotations(xml_dir)
    N = int(training_ratio * len(annotations))
    keys = list(annotations.keys())
    # shuffle the filenames to get a random training set
    random.shuffle(keys)
    # open the files to write the assignments to
    with open(classname1 + '_train.txt', 'w') as training_file, open(
        classname1 + '_test.txt', 'w'
    ) as test_file, open('test.txt', 'w') as test, open('trainval.txt', 'w') as trainval:
        for i, filename in enumerate(keys):
            # write the first N files to the training set
            if i < N:
                trainval.write(filename + '\n')
                # write 1 if the image contains the positive class, else -1
                if classname1 in annotations[filename]:
                    training_file.write(filename + '  1\n')
                elif classname2 in annotations[filename]:
                    training_file.write(filename + '  1\n')
                else:
                    training_file.write(filename + ' -1\n')
            # the rest of the files go to the test set, which all get 0s
            else:
                test.write(filename + '\n')
                test_file.write(filename + ' 0\n')
