'''
TODO: explaine filili nested lists of dics concept (pure python)
TODO: make importers, exporters for those lists (table, etc.)
TODO: explain difference between Sherpa model list and filili modellist
TODO: Mark consistenly where shepar models and where fililiy modes are required
I started the convenciotn that fililit models (nested lists) are required
the parameter names start with f, but that's no consistent so far.
'''
from warnings import warn

from .shmodelshelper import copy_pars, set_parameter_from_dict
from .utils import get_flat_elements

from sherpa import stats, optmethods, fit

# linelist = [[group of model 1], [group of models 2], [line 3], [line 4]]
# Note: Currently even cingle models must be in a list of one element.
# Will lift that restriction later.
# each line must have "name"
# group of lines have fixed wavelength, same modeltype


class NonUniqueModelNames(Exception):
    pass


def constant_difference(modellist, parameter):
    '''Link models so that the difference in a certain parameter stays constant.

    Parameters
    ----------
    modellist : list
        Elements of the list must be Sherpa model instances that all have
        ``parameter``. The do not have to be of the same type, e.g. all line
        profiles have a parameter named ``pos``.
    parameter : string
        name of parameter

    Example
    -------

    >>> from sherpa.astro.models import Lorentz1D
    >>> m1 = Lorentz1D('m1')
    >>> m2 = Lorentz1D('m2')
    >>> m1.pos=4
    >>> m2.pos=6
    >>> constant_difference([m1, m2], 'pos')
    >>> m1.pos = 5.
    >>> print m2.pos.val
    7.0

    '''
    basepar = getattr(modellist[0], parameter)
    for m in modellist[1:]:
        diff = getattr(m, parameter).val - basepar.val
        m.pos.link = basepar + diff


class ModelMaker(object):
    _modelcounter = 0

    @property
    def modelcounter(self):
        '''Return an unused number for labelling models.

        The number retunred is unique over all instances of this class.
        '''
        self.__class__._modelcounter += 1
        return self._modelcounter

    def check_names_complete(self, modellist):
        '''Check if every model in the list has a 'name' specificed.

        Parameters
        ----------
        modellist : list
            nested list of lists. Innermost eement are dicts.

        Returns
        -------
        complete : bool
        '''
        return None in get_flat_elements(modellist, 'name')

    def check_names_unique(self, modellist):
        '''Raise an Error if non-unique names are found in the modellist.'''
        nameslist = list(get_flat_elements(modellist, 'name', include_missing=False))
        nonunique = set([x for x in nameslist if nameslist.count(x) > 1])
        if len(nonunique) > 0:
            raise NonUniqueModelNames('Lines names must be unique. The following names appear more than once: {0}'.format(nonunique))

    def make_model_group(self, fmodels, basemodel):
        '''Make a group of sherpa model instances based on the same default values.

        Parameter
        ---------
        fmodels : list of dicts
            Each dict in the list describes one Sherpa model. Not all possible
            parameter values have to be given, e.g.
            ``[{'pos.val': 123., 'pos.min': 100.}]`` for Gaussian1D model.
        basemodel : sherpa model instance
            All returned models are of the same type as basemodel. Additionally,
            the parameters in basemodel (e.g. val, min, max, link) are used as
            defaults for the generated models.

        Returns
        -------
        modelgroup : list
            The elements in this list are sherpa models
        '''
        self.check_names_unique(fmodels)

        modelgroup = []
        for l in fmodels:
            if 'name' in l:
                name = l['name']
            else:
                name = 'mdl{0}'.format(self.modelcounter)
            newmodel = basemodel.__class__(name)
            copy_pars(basemodel, newmodel)
            # FIXME: Problems are possible when the current max, min range is not
            # large enough to set the max. min to the new numbers.
            # Not likely to be a problem, because default ranges are large.
            for par in newmodel.pars:
                set_parameter_from_dict(par, l)
            modelgroup.append(newmodel)
        return modelgroup

    def models_from_list(self, fmodellist, basemodellist):
        '''Make a list of additive Sherpa models.

        Parameters
        ----------
        fmodellist : list of lists
            Each list in fmodellist has to contain a dict that describes a
            model
        basemodellist : list
            Each element in basemodel is a Sherpa model instance, that serves
            as the base model for all dicts in the respective list.

        Returns
        -------
        modellist : list of lists
            Sherpa model instances corresponding to the input in fmodellist

        Example
        ------
        In this example, we make a model with a constant (initialized at the
        default parameter values) and two Lorentzian line.
        Both are fairly narrow and they starting positions are 2. and 3.
        In this example, we could have types the model out just as easily by
        hand, but when we manage a dozen or more lines of very similar
        properties, then this function will come in handy.

        >>> from sherpa.astro.models import Lorentz1D
        >>> from sherpa.models import Const1D
        >>> from filili import ModelMaker
        >>> maker = ModelMaker()
        >>> linebase = Lorentz1D()
        >>> linebase.fwhm = 0.01
        >>> constbase = Const1D()
        >>> myguess = [[{'name': 'const'}],
        ...            [{'name': 'line1', 'pos.val':2.},
        ...             {'name': 'line2', 'pos.val': 3.}]]
        >>> mlist = maker.models_from_list(myguess, [constbase, linebase])
        >>> mall = maker.finalize_model(mlist)
        >>> print mall
        ((const + line1) + line2)
        Param        Type          Value          Min          Max      Units
        -----        ----          -----          ---          ---      -----
        const.c0     thawed            1 -3.40282e+38  3.40282e+38
        line1.fwhm   thawed         0.01            0  3.40282e+38
        line1.pos    thawed            2 -3.40282e+38  3.40282e+38
        line1.ampl   thawed            1 -3.40282e+38  3.40282e+38
        line2.fwhm   thawed         0.01            0  3.40282e+38
        line2.pos    thawed            3 -3.40282e+38  3.40282e+38
        line2.ampl   thawed            1 -3.40282e+38  3.40282e+38
        '''
        if len(fmodellist) != len(basemodellist):
            raise ValueError('There needs to be one base model for every (group of) models.')

        self.check_names_unique(fmodellist)

        modellist = []
        for ms, basemodel in zip(fmodellist, basemodellist):
            modelgroup = self.make_model_group(ms, basemodel)
            modellist.append(modelgroup)

        return modellist

    def finalize_model(self, modellist):
        '''Construct a single Sherpa model from a list of models.

        Overwrite this method in the derived class, e.g. to wrap the
        entire model into a convolution model to apply an instrument
        response function or to link the wavelength of all lines in
        the first group of elements.
        The default is to just add up all the models in the input
        ``modellist``.

        Parameters
        ----------
        modellist : list of lists of Sherpa models
            Usually, the model will be a model instance that contains the
            sum of several spectral line models.

        Returns
        -------
        model : Sherpa model
            A Sherpa model instance
        '''
        flatlist = [item for sublist in modellist for item in sublist]
        model = flatlist[0]
        for m in flatlist[1:]:
            model = model + m
        return model


class Fitter(object):
    fit_settings = {'stat': stats.Chi2(),
                    'method': optmethods.LevMar(),
                    'estmethod': None,
                    'itermethod_opts': {'name': 'none'},
                    }

    def fit_valid(self, fitresult):
        '''Decide if the resulting fit is a valid result.

        You might want to modify this method in a derived class to analyze
        if a fit should be accepted, based on e.g. the value of the fit
        statistic.

        In this implementation, we simply return the value of `succeed`
        reported by Sherpa.

        Parameters
        ----------
        fitresult : Sherpa fit results instance

        Returns
        -------
        valid : bool
        '''
        return fitresult.succeeded

    def fit(self, data, model):
        '''Perform fit.

        This fit uses the fit parameters set in `fit_settings`, so an easy way
        to customize the way the fits is done is to change that attribute.

        Override this method for more complex fitting schemes.
        Currently, it just performs a single Sherpa fit with default settings.
        Other things that could be implemented here is to restart a fit with
        different initial values or with a different fit method if the results
        seem unrealistic.

        Parameters
        ----------
        data : sherpa data
            This must be an instance of a Sherpa data class, e.g Data1D,
            with all the appropriate notice and ignore filters already set.
        model : sherpa model
            Sherpa model instance.

        Returns
        -------
        result : Sherpa fit result
        '''
        f = fit.Fit(data, model, **self.fit_settings)
        result = f.fit()
        if not self.fit_valid(result):
            warn('Fit failed.')
        errres = f.est_errors()
        return result, errres


class Master(object):

    def __init__(self, modelmaker, fitter, plotter, fitreporter, confreporter):
        self.modelmaker = modelmaker
        self.fitter = fitter
        self.plotter = plotter
        self.fitreporter = fitreporter
        self.confreporter = confreporter

    def loop_regions(self, data, regions, basemodellist=None):
        '''

        Parameters
        ----------
        regions : list
            Every element in the list describes a spectral region that needs to
            be fit. The elements are dictionaries which need to have the
            following keys:

            - *optional* ``name`` : string name to identify the region
            - ``fmodellist``: list. See `models_from_list` for details.
              Each element in the list describes models to be fitted to one
              region. Each element is again a list or lists,  where each
              element describes a group of models
            - *optional* ``basemodels``: list. See `models_from_list`
              for details.

        basemodellist : list or ``None``
            List of Sherpa model instances to be used as base models for
            regions, that do not have a ``basemodels`` key. See
            `models_from_list` for details.
        '''
        for region in regions:
            if 'basemodels' in region:
                basemodels = region['basemodels']
            elif basemodellist is not None:
                basemodels = basemodellist
            else:
                raise ValueError('Base models must be defined in region or as basemodellist.')
            fmodellist = region['fmodellist']
            data.notice(region['range'][0], region['range'][1])

            modellist = self.modelmaker.models_from_list(fmodellist, basemodels)
            model = self.modelmaker.finalize_model(modellist)

            fitresult, uncertainty = self.fitter.fit(data, model)
            self.plotter.plot(data, model, title=region['name'])
            self.fitreporter.report(fitresult, region)
            self.confreporter.report(uncertainty, region)


# from COSlsf import empG160M


# class COSFUVModelMaker(ModelMaker):
#     def finalize_model(self, modellist):
#         model = super(COSFUVModelMaker, self.finalize_model(modellist))

#         # Assuming H2 lines are first sublist
#         constant_difference(modellist[0], 'pos')

#         return empG160M(model)
