import os
import numpy as np
import sherpa.astro.ui as ui

try:
    import pychips
    has_chips = True
except ImportError:
    has_chips = False

try:
    import matplotlib.pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False

class NoPlottingSystemError(Exception):
    pass


import filili
import filili.multilist

#ui.set_stat('cstat')
ui.set_method_opt('epsfcn', 2.2e-16)


#ui.set_model('gauss1d.baseline')
#baseline = ui.get_model_component('baseline')
#baseline.ampl.min = 0.
#baseline.fwhm.max = .2
#baseline.fwhm.min = .01
#baseline.fwhm = .01
#baseline.fwhm.freeze()


def fit_lines(linelist, id = None, delta_lam = .2, plot = False, outfile = None):
    mymodel = filili.multilinemanager.GaussLines('const1d', id = id, baseline = baseline)
    linelist['fililiname'] = [''] * len(linelist['linename'])
    for i in range(len(linelist['linename'])):
        lname = linelist['linename'][i]
        lwave = linelist['wave'][i]
        previouslines = set(mymodel.line_name_list())
        mymodel.add_line(linename = filter(lambda x: x.isalnum(), lname), pos = lwave)
        linenamelist = mymodel.line_name_list()
        newline = (set(linenamelist) - previouslines).pop()
        linelist['fililiname'][i] = newline
        if i ==0:
            firstline = linenamelist[0]
            ui.get_model_component(firstline).pos.max = lwave + delta_lam/10.
            ui.get_model_component(firstline).pos.min = lwave - delta_lam/10.
        else:
            dl = lwave - linelist['wave'][0]
            ui.link(ui.get_model_component(newline).pos, ui.get_model_component(firstline).pos + dl)
    
    #ui.set_analysis("wave")
    ui.ignore(None, None) #ignores all data
    ui.notice(min(linelist['wave'])-delta_lam, max(linelist['wave']) + delta_lam)
    ui.fit(id)
    if plot:
        ui.plot_fit(id)
        if has_chips:
            pychips.set_curve("crv1",["err.*","true"])
            pychips.set_plot_title(linelist['name'])
            if outfile is not None:
                pychips.print_window(outfile, ['clobber','true'])
        elif has_mpl:
            plt.title(linelist['name'])
            if outfile is not None:
                plt.savefig(outfile)
        else:
            raise NoPlottingSystemError("Neither pychips nor matplotlib are found.")


def fit_multiplets(multipletlist, id = None, outpath = None, plot = False, delta_lam = .2):
    #
    n_lines =  np.sum([len(mult['wave']) for mult in multipletlist])
    result = np.zeros(n_lines, dtype = {'names': ['multname', 'linename', 'wave', 'flux', 'errup', 'errdown', 'photons', 'photonflux'], 'formats': ['S30', 'S30', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4']})
    currentline = 0
    #
    for mult in multipletlist:
        if outpath is not None:
            outfile = os.path.join(outpath, filter(lambda x: x.isalnum(), mult['name']))
        else:
            outfile = None
        fit_lines(mult, id, delta_lam = delta_lam, plot = plot, outfile = outfile)
        #    
        ui.conf(id)
        conf_res = ui.get_conf_results()
        source = ui.get_source(id)
        #set all ampl to 0 and only for 1 line to real value
        set_all_val('c0', 0., id)
        for lname, lfili, lwave in zip(mult['linename'], mult['fililiname'], mult['wave']):
            print 'Fitting line '+str(currentline+1)+'/'+str(n_lines)
            par = ui.get_model_component(lfili)
            indconf_res = ( np.array(conf_res.parnames) == lfili+'.ampl').nonzero()[0]
            set_all_val('ampl', 0., id)
            
            par.ampl.val = conf_res.parvals[indconf_res]
            counts = ui.calc_model_sum(None, None, id)
            #
            print(lname, counts)
            photonflux = ui.calc_photon_flux(None, None, id)
            # determine scaling between ampl and flux
            par.ampl.val = 1
            amp2flux = ui.calc_energy_flux(None, None, id)

            par.ampl.val = conf_res.parvals[indconf_res]
            #
            val = conf_res.parvals[indconf_res] * amp2flux
            errdown = conf_res.parmins[indconf_res] * amp2flux if conf_res.parmins[indconf_res] else np.nan
            errup  = conf_res.parmaxes[indconf_res] * amp2flux if conf_res.parmaxes[indconf_res] else np.nan
            #
            result[currentline] = (mult['name'], lname, lwave, val, errup, errdown, counts, photonflux)
            #
            currentline +=1
    return result

def set_all_val(name, val, id = None):
    'e.g. set_all_val("ampl", 0)'
    source = ui.get_source(id)
    for par in source.pars:
        if par.name == name:
            par.val = val