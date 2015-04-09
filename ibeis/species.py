from __future__ import absolute_import, division, print_function
import utool as ut
from ibeis import constants as const
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[species]', DEBUG=False)


def species_has_detector(species_text):
    return species_text in const.SPECIES_WITH_DETECTORS


def get_working_species_set():
    """ hack to make only species with detectors show up """
    # TODO: FUNCTIONS SHOULD NOT BE IN CONSTANTS
    # TODO: allow for custom user-define species
    #RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = not ut.get_argflag('--allspecies')
    RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = ut.get_argflag('--no-allspecies')
    if RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS:
        working_species_tups = [
            (species_tup.species_nice, species_tup.species_text)
            for species_tup in const.SPECIES_TUPS
            if species_has_detector(species_tup.species_text)
        ]
    else:
        working_species_tups = [
            (species_tup.species_nice, species_tup.species_text)
            for species_tup in const.SPECIES_TUPS
        ]
    return working_species_tups
