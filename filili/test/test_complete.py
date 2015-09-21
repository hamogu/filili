import numpy as np
import pytest

from sherpa.data import Data1D
from sherpa.models import Const1D, Gauss1D

from ..low_fit import ModelMaker, Fitter, Master, SherpaReporter

def gauss(x, x0, sigma):
    return 1./np.sqrt(2 * np.pi) / sigma * np.exp(-(x - x0)**2/(2. * sigma**2))


@pytest.fixture
def spectrum():
    lines = {'x0': [20., 25., 40., 60.],
             'norm': [8., 3., 6., 30.],
             'sigma': [3., 1., 2., 5.]}

    x = np.arange(1., 100., .5)
    y = 1.
    for i in range(len(lines['x0'])):
        y += lines['norm'][i] * gauss(x, lines['x0'][i], lines['sigma'][i])
    y += np.random.normal(loc=0., scale=.05, size=len(x))
    err = .1 * np.ones_like(y)
    return {'lines': lines, 'x': x, 'y': y, 'y_err': err}

@pytest.fixture
def sherpaspec():
    spec = spectrum()
    return Data1D('test', spec['x'], spec['y'], spec['y_err'])


def test_fitter(sherpaspec):
    '''Test that the interface to the fitter works.

    (We assume that the fitter itself is fine, that should be tested by the
    package that we use as the fitting backend.)
    '''
    sherpaspec.notice(50., 70.)
    model = Const1D() + Gauss1D()
    model.parts[1].pos.val = 58.  # set a reasonable starting guess
    fitter = Fitter()
    result, fit = fitter.fit(sherpaspec, model)
    assert np.allclose(result.parvals, [1., 2.355 * 5., 60., 30./5. /np.sqrt(2.*np.pi)], rtol=.1)


class Test_sherpa_fits(object):

    def test_fit_regions(self, sherpaspec):
        constbase = Const1D()
        gaussbase = Gauss1D()
        guess = [{'name': 'region1', 'range': [15., 30.],
                  'fmodellist': [[{'name': 'const'}],
                                 [{'name': 'line1', 'pos.val': 20.5},
                                  {'name': 'line2', 'pos.val': 26.5} ]]},
                 {'name': 'region2', 'range': [35., 45.],
                  'fmodellist': [[{'name': 'const'}],
                                 [{'name': 'line3', 'pos.val': 39.}]]}
        ]

        reporter = SherpaReporter()
        fitmaster = Master(ModelMaker(), Fitter(), reporter, reporter)
        fitmaster.loop_regions(sherpaspec, guess, basemodellist=[constbase, gaussbase])
        for r in guess:
            # Check is fit is good.
            # If it is, the input was read correctly.
            assert reporter.results[r['name']]['rstat'] < 1.
