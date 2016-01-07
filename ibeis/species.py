# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from ibeis import constants as const

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[species]')

species_mapping = {
    'bear_polar'          :       ('PB', 'Polar Bear'),
    'building'            : ('BUILDING', 'Building'),
    'cheetah'             :     ('CHTH', 'Cheetah'),
    'elephant_savanna'    :     ('ELEP', 'Elephant (Savanna)'),
    'frog'                :     ('FROG', 'Frog'),
    'giraffe_masai'       :     ('GIRM', 'Giraffe (Masai)'),
    'giraffe_reticulated' :      ('GIR', 'Giraffe (Reticulated)'),
    'hyena'               :    ('HYENA', 'Hyena'),
    'jaguar'              :      ('JAG', 'Jaguar'),
    'leopard'             :     ('LOEP', 'Leopard'),
    'lion'                :     ('LION', 'Lion'),
    'lionfish'            :       ('LF', 'Lionfish'),
    'lynx'                :     ('LYNX', 'Lynx'),
    'nautilus'            :     ('NAUT', 'Nautilus'),
    'other'               :    ('OTHER', 'Other'),
    'rhino_black'         :   ('BRHINO', 'Rhino (Black)'),
    'rhino_white'         :   ('WRHINO', 'Rhino (White)'),
    'seal_saimma_ringed'  :    ('SEAL2', 'Seal (Siamaa Ringed)'),
    'seal_spotted'        :    ('SEAL1', 'Seal (Spotted)'),
    'snail'               :    ('SNAIL', 'Snail'),
    'snow_leopard'        :    ('SLEOP', 'Snow Leopard'),
    'tiger'               :    ('TIGER', 'Tiger'),
    'toads_wyoming'       :   ('WYTOAD', 'Toad (Wyoming)'),
    'water_buffalo'       :     ('BUFF', 'Water Buffalo'),
    'wildebeest'          :       ('WB', 'Wildebeest'),
    'wild_dog'            :       ('WD', 'Wild Dog'),
    'whale_fluke'         :       ('WF', 'Whale Fluke'),
    'whale_humpback'      :       ('HW', 'Humpback Whale'),
    'whale_shark'         :       ('WS', 'Whale Shark'),
    'zebra_grevys'        :       ('GZ', 'Zebra (Grevy\'s)'),
    'zebra_hybrid'        :       ('HZ', 'Zebra (Hybrid)'),
    'zebra_plains'        :       ('PZ', 'Zebra (Plains)'),
    const.UNKNOWN         :  ('UNKNOWN', 'Unknown'),
}
