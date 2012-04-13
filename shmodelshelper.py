"""
Module with helpers to manage Sherpa model parameters
"""

import json
import os
from sherpa.astro.ui import get_model_component, get_model
from sherpa.utils.err import IdentifierErr

def save_pars(filename, modcomps=[], clobber=False):
    """
    Save Sherpa model parameter attributes to an ASCII file

    `filename`  ASCII file name
    `modcomps`  list of model components (strings or objects) to save
    `clobber`   clobber the file if it exists
    
    :author: Brian Refsdal
    
    Example:

        from sherpa.astro.ui import *
        from save_pars import save_pars, load_pars

        set_model(gauss1d.g1 + gauss1d.g2)

        ... set up parameters, fit

        save_pars('mypars.out', [g1, g2])
        or
        save_pars('mypars.out', list_model_components(), clobber=True)
        load_pars('mypars.out', [g1, g2])
    """

    if not isinstance(filename, basestring):
        raise TypeError("filename '%s' is not a string" % str(filename))

    clobber = bool(clobber)
    if os.path.isfile(filename) and not clobber:
        raise ValueError("file '%s' exists and clobber is not set" %
                         str(filename))

    saved = {}
    for comp in modcomps:
        for par in get_model_component(comp).pars:
            for elem in ["val", "min", "max"]:
                key = par.fullname + "." + elem
                saved[key] = getattr(par,elem)

            elem = "frozen"
            key = par.fullname + "." + elem
            saved[key] = int(getattr(par,elem))

            elem = "link"
            key = par.fullname + "." + elem
            attr = getattr(par,elem)
            if attr:
                saved[key] = str(attr.fullname)

    fd = file(filename, 'w')
    fd.write(json.dumps(saved))
    fd.close()


def load_pars(filename, modcomps=[]):
    """
    Load Sherpa model parameter attributes from an ASCII file
    and set the input model components with the parameter attributes.

    `filename`  ASCII file name
    `modcomps`  list of model components (strings or objects) to save
    
    :author: Brian Refsdal
    
    See `save_pars` for an example.
    
    """

    if not isinstance(filename, basestring):
        raise TypeError("filename '%s' is not a string" % str(filename))
    
    if not os.path.isfile(filename):
        raise IOError("file '%s' does not exist" % str(filename))

    fd = open(filename, 'r')
    saved = json.loads(fd.readline().strip())
    fd.close()

    for comp in modcomps:
        for par in get_model_component(comp).pars:
            for elem in ["val", "min", "max"]:
                key = par.fullname + "." + elem
                if saved.has_key(key):
                    setattr(par, elem, saved.get(key))

            elem = "frozen"
            key = par.fullname + "." + elem
            if saved.has_key(key):
                setattr(par, elem, bool(int(saved.get(key))))

            elem = "link"
            key = par.fullname + "." + elem
            attr = str(saved.get(key, ''))
            if attr:
                mdl, param = attr.split('.')
                param = getattr(get_model_component(mdl), param)
                setattr(par, elem, param)


def copy_pars(oldcomp, newcomp, sametype = True):
    """copy parameters from one component to an onther
    
    Both components need to be of the same type, e.g. both are gaus1d models
    This routine then copies `val`, `max`, `min`, `frozen` and `link` values.
    
    Example:
    >>> set_model(gauss1d.g1 + gauss1d.g2)
    >>> g1.pos.min = 0.
    >>> copy_pars(g1, g2)
    
    Parameters
    ----------
    :param oldcomp: Sherpa model component
        component with original values
    :param newcomp: Sherpa model component
        values of this component will be set
        
    TBD: replace get_model_component(oldcomp).pars with some way that iterates over names, so that parameters can be copied between two line types, even if pos is once the first and once the second parameter. 
    """
    if sametype:
        if not (type(oldcomp) == type(newcomp)):
            raise TypeError('Old and new model component must be of same type')
    #
    for parold, parnew in zip(get_model_component(oldcomp).pars, get_model_component(newcomp).pars):
        # min cannot be above max.
        # set to -+inf to avoid problems with previously set pars
        setattr(parnew, "min", getattr(parnew, "hard_min"))
        setattr(parnew, "max", getattr(parnew, "hard_max"))
        for elem in ["min", "max", "val", "frozen", "link"]:
            setattr(parnew, elem, getattr(parold,elem))

def get_model_parts(id = None):
    '''obtain a list of strings for sherpa models
    
    Iterate through all components which are part of the Sherpa model
    and return their identifiers. Ignore all composite models.
    
    Example:
    >>> from sherpa.ui import *
    >>> set_model('const1d.c + gauss1d.lineg1 + gauss1d.lineg2 + gauss1d.lineg3')
    >>> show_model()
        Model: 1
        (((const1d.c + gauss1d.lineg1) + gauss1d.lineg2) + gauss1d.lineg3)
    >>> get_model_parts()
        ['c', 'lineg1', 'lineg2', 'lineg3']

    '''
    try:
        return set([par.modelname for par in get_model(id).pars])
    except IdentifierErr:
        return set([])