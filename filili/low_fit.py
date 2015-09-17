from warnigs import warn

from .shmodelshelper import copy_pars

from sherpa.astro.ui import set_parameter_from_dict
from sherpa import stats, optmethods, fit

# linelist = [[group of model 1], [group of models 2], [line 3], [line 4]]
# Note: Currently even cingle models must be in a list of one element.
# Will lift that restriction later.
# each line must have "name"
# group of lines have fixed wavelength, same modeltype


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
    >>>print m2.pos.val
    7.0

    '''
    basepar = getattr(modellist[0], parameter)
    for m in modellist[1:]:
        diff = getattr(m, parameter).val - basepar.val
        m.pos.link = basepar + diff


class MasterFitter(object):

    fit_settings = {'stat': stats.Chi2(),
                    'method': optmethods.LevMar(),
                    'estmethod': None,
                    'itermethod_opts': {'name': 'none'},
                    }

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
        # FIXME Add check that names are unique. Not strictly required for low-level interface
        # but will cause massive confusion at ui level.
        modelgroup = []
        for l in fmodels:
            newmodel = basemodel.__class__(l['name'])
            copy_pars(basemodel, newmodel)
            # FIXME: Problems are possible when the current max, min range is not
            # large enough to set the max. min to the new numbers.
            # Not likely to be a problem, because default ranges are large.
            for par in newmodel.pars:
                set_parameter_from_dict(par, l)
            modelgroup.append(newmodel)
        return modelgroup

    def additive_models_from_list(self, fmodellist, basemodellist):
        '''Make a list of additive Sherpa models.

        Parameters
        ----------
        fmodellist : list of lists
            Each list in fmodellist has to contain a dict that describes a model
        basemodellist : list
            Each element in basemodel is a Sherpa model instance, that serves as the base
            model for all dicts in the respective list.

        Returns
        -------
        modellist : list of lists
            Sherpa model instances corresponding to the input in fmodellist
        all_model : sherpa model instance
            A model that adds all the models in modellist together.

        Example
        ------
        In this example, we make a model with a constant (initialized at the default parameter values)
        and two Lorentzian line. Both are fairly narrow and they starting positions are 2. and 3.
        In this example, we could have types the model out just as easily by hand, but when we
        manage a dozen or more lines of very similar properties, then this function will
        come in handy.

        >>> from sherpa.astro.models import Lorentz1D
        >>> from sherpa.models import Const1D
        >>> linebase = Lorentz1D()
        >>> linebase.fwhm = 0.01
        >>> constbase = Const1D()
        >>> myguess = [[{'name': 'const'}],
        ...            [{'name': 'line1', 'pos.val':2.},
        ...             {'name': 'line2', 'pos.val': 3.}]]
        >>> mlist, mall = additive_models_from_list(myguess, [constbase, linebase])
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

        # FIXME Add check that names are unique. Not strictly required for low-level interface
        # but will cause massive confusion at ui level.
        modellist = []
        for ms, basemodel in zip(fmodellist, basemodellist):
            modelgroup = make_model_group(ms, basemodel)
            modellist.append(modelgroup)

        flatlist = [item for sublist in modellist for item in sublist]
        all_model = flatlist[0]
        for m in flatlist[1:]:
            all_model = all_model + m

        return modellist, all_model

    def wrap_model(self, model):
        '''Modify the additive line model.

        Overwrite this method in the derived class, e.g. to wrap the
        entire model into a convolution model to apply an instrument
        response function.
        The default is to just return the additive model unchanged.

        Parameters
        ----------
        model : Sherpa model
            Usually, the model will be a model instance that contains the
            sum of several spectral line models.

        Returns
        -------
        model : Sherpa model
            A Sherpa model instance
        '''
        return model

       sherpa.fit.Fit(self, data, model, stat=None, method=None, estmethod=None, itermethod_opts={'name': 'none'})
       stat = stats.Chi2()

    def loop_regions(self, regions, basemodellist=None):
        '''

        Parameters
        ----------
        regions : list
            Every element in the list describes a spectral region that needs to be fit.
            The elements are dictionaries which need to have the following keys:

            - ``name`` : string name to identify the region
            - ``fmodellist``: list. See `additative_models_from_list` for details.
              Each element in the list describes models to be fitted to one region.
              Each element is again a list or lists,  where each element describes a group of models
            - *optional* ``basemodels``: list. See `additative_models_from_list` for details.

        basemodellist : list or ``None``
            If the basemodels are the same for every region,
        for region in regions:
            if basemodellist is None:
                basemodels = region['basemodels']
            else:
                basemodels = basemodellist
            fmodellist = region['fmodellist']
            data.notice(region['range'][0], region['range'][1])

            modellist, all_model = additive_models_from_list(fmodellist, basemodellist)
            model = self.wrap_model(all_model)
            constant_difference(modellist[0], 'pos')  # Assuming H2 lines are fist sublist


            f = fit.Fit(data, model, **self.fit_settings)
            result = f.fit()
            if not result.succeeded:
                warn('Fit failed')
            # check chi^2 etc.
            # plot_fit
            errres = f.est_errors()

        return some thing


class COSFUVFitter(Masterfitter):
    def wrap_model(self, model):
        return empG160M(model)
