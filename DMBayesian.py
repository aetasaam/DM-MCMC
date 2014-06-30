#   ______ ___  ___     ___  ___ _____ ___  ___ _____                               #
#   |  _  \|  \/  |     |  \/  |/  __ \|  \/  |/  __ \                              #
#   | | | || .  . |     | .  . || /  \/| .  . || /  \/                              #
#   | | | || |\/| |     | |\/| || |    | |\/| || |                                  #
#   | |/ / | |  | |     | |  | || \__/\| |  | || \__/\                              #
#   |___/  \_|  |_/     \_|  |_/ \____/\_|  |_/ \____/                              #
#####################################################################################
##  Bayesian analysis code for producing posteriors using emcee - The MCMC Hammer, ##
##  a pure-Python implementation of Goodman & Weare's Affine Invariant Markov      ##
##  chain Monte Carlo (MCMC) Ensemble sampler.                                     ##
##                                                                                 ##
##  Requires the emcee package found at dan.iel.fm/emcee/current/                  ##
##                                                                                 ##
##  Author - Cedric Flamant                                                        ##
#####################################################################################

import sys
import numpy as np
import scipy as sp
import ProgressBar as progbar
from scipy.misc import factorial
import matplotlib.pyplot as plt
import emcee
from scipy.stats import poisson

# All relevant parameters to run the script can be modified in this main method
def main():
    read_data("DMm400AVu_cf.txt")
    set_prior(uniform)
    ndim = 3
    nwalkers = 90   # Number of walkers. Generally more is better. Must be even.
    burnlinks = 300 # Use this to set a burn-in time for the MCMC. 300 seems good.
    links = 10000   # Make this larger to get more Monte Carlo samples 
    hist_bincnt = 300 # Number of bins for the plotted histogram. No effect on computation.
    # run_MCMC_onebin(4, nwalkers, burnlinks, links, hist_bincnt) # Posterior for a single bin
    run_MCMC(nwalkers, burnlinks, links, hist_bincnt) # Posterior for all bins


#####################################################################################
#                ______                     __   _                                  #
#               / ____/__  __ ____   _____ / /_ (_)____   ____   _____              #
#              / /_   / / / // __ \ / ___// __// // __ \ / __ \ / ___/              #
#             / __/  / /_/ // / / // /__ / /_ / // /_/ // / / /(__  )               #
#            /_/     \__,_//_/ /_/ \___/ \__//_/ \____//_/ /_//____/                #
#####################################################################################
                                                         

#################################################
#   Global Variables                            #
#################################################
# dummy prior
def dummy(param):
    return 0
N_bkg = 0.0     # Model backgroud counts (avg. b)
E_bkg = 0.0     # Systematic error on background
N_sig = 0.0     # Model signal counts (avg. s)
E_sig = 0.0     # Systematic error on signal
Data_obs = 0.0  # Observed total counts
prior = dummy   # Current prior function being used


#################################################
#   Read File Method                            #
#################################################

# Takes file path and saves variables
def read_data(fname):
    global N_bkg, E_bkg, N_sig, E_sig, Data_obs
    N_bkg, E_bkg, N_sig, E_sig, Data_obs = np.loadtxt(fname, skiprows=1, unpack = True)

#################################################
#   Prior Function Definitions                  #
#   Note that param = (eta, s, b)               #
#   where eta is a single value and s and b     #
#   are arrays                                  #
#################################################

# Uniform prior
def uniform(param):
    return 1

# Jeffreys prior
def jeffreys(param):
    lam = param[0]*param[1] + param[2]
    return np.sqrt(np.sum(param[1]/lam))

#################################################
#   Set Prior Function                          #
#################################################
def set_prior(chosen_prior):
    global prior
    prior = chosen_prior


#################################################
#   Log Likelihoods                             #
#   These are passed to the MCMC to get the     #
#   posterior.                                  #
#################################################

# Log Likelihood of getting all the data in all bins.
# The xs and xb integration variables are treated as extra dimensions.
# Marginalizing over these variables using the output MCMC chains is
# the same as performing multidimensional integration over their ranges.
# Note: pos = (eta, xs, xb)
def log_likelihood_all(pos):
    eta = pos[0]
    xs = pos[1]
    xb = pos[2]
    if eta < 0.:
        return np.NINF
    s = N_sig * (1 + E_sig/N_sig)**xs
    b = N_bkg * (1 + E_bkg/N_bkg)**xb
    lam = eta * s + b
    log_Normal_xs = np.log(1./np.sqrt(2*np.pi)) -xs**2/2.
    log_Normal_xb = np.log(1./np.sqrt(2*np.pi)) -xb**2/2.
    log_Poisson_lam = np.log(poisson.pmf(Data_obs, lam))
    log_likelihood = np.log(prior((eta,s,b))) + np.sum(log_Poisson_lam) + log_Normal_xs + log_Normal_xb
    progbar.step_progress()
    return log_likelihood

# Log Likelihood of getting all the data in all bins.
# Note: pos = (eta, xs, xb)
# (Legacy) This calculation is technically more exact, but it increases computation 
# time significantly (factor of 100!). I could not discern a difference.
def log_likelihood_all_legacy(pos):
    eta = pos[0]
    xs = pos[1]
    xb = pos[2]
    if eta < 0.:
        return np.NINF
    s = N_sig * (1 + E_sig/N_sig)**xs
    b = N_bkg * (1 + E_bkg/N_bkg)**xb
    lam = eta * s + b
    log_Normal_xs = np.log(1./np.sqrt(2*np.pi)) -xs**2/2.
    log_Normal_xb = np.log(1./np.sqrt(2*np.pi)) -xb**2/2.
    log_factorial = np.zeros_like(Data_obs)
    for i in range(0,len(log_factorial)):
        for j in range(1,np.round(Data_obs[i]).astype(np.int)):
            log_factorial[i] += np.log(j)
    log_Poisson_lam = Data_obs*np.log(lam) -lam - log_factorial
    log_likelihood = np.log(prior((eta,s,b))) + np.sum(log_Poisson_lam) + log_Normal_xs + log_Normal_xb 
    progbar.step_progress()
    return log_likelihood

# Log Likelihood of getting all the data in a single bin
# Note: pos = (eta, xs, xb)
def log_likelihood_onebin(binnum, pos):
    eta = pos[0]
    xs = pos[1]
    xb = pos[2]
    if eta < 0.:
        return np.NINF
    s = N_sig[binnum] * (1 + E_sig[binnum]/N_sig[binnum])**xs
    b = N_bkg[binnum] * (1 + E_bkg[binnum]/N_bkg[binnum])**xb
    lam = eta * s + b
    log_Normal_xs = np.log(1./np.sqrt(2*np.pi)) -xs**2/2.
    log_Normal_xb = np.log(1./np.sqrt(2*np.pi)) -xb**2/2.
    log_Poisson_lam = np.log(poisson.pmf(Data_obs[binnum], lam))
    log_likelihood = np.log(prior((eta,s,b))) + log_Poisson_lam + log_Normal_xs + log_Normal_xb
    progbar.step_progress()
    return log_likelihood

#################################################
#   Various plot types:                         #
#   Single plot of all the bins,                #
#   Single plot of one bin posterior            #
#################################################

# Posterior for all the bins
## Takes number of walkers, number of links to burn, and number of links
def run_MCMC(nwalkers, burnlinks, links, hist_bincnt):
    ndim = 3
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_all)
    progbar.set_totalsteps(float(burnlinks)*nwalkers)
    print 'Begin burn in'
    pos, prob, state = sampler.run_mcmc(p0,burnlinks)
    progbar.update_progress(1)
    print 'Burn in completed'
    sampler.reset()
    progbar.set_totalsteps(float(links)*nwalkers)
    sampler.run_mcmc(pos, links)
    progbar.update_progress(1)
    print('Main chain completed')
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    # Determine the 95% confidence level by sorting the marginalized chain and finding the 
    # eta element that is 95% of the way down the sorted chain
    sorted_eta = np.sort(sampler.flatchain[:,0])
    conf95_index = np.round(0.95*len(sorted_eta) - 1).astype(np.int)
    conf998_index = np.round(0.998*len(sorted_eta) - 1).astype(np.int)
    conf95 = sorted_eta[conf95_index]
    conf998 = sorted_eta[conf998_index]
    print '95% upper limit: ' + str(conf95)
    plt.figure()
    y, x, o = plt.hist(sampler.flatchain[:,0], hist_bincnt, range=[0,conf998], histtype="step", normed = 1)
    yciel = 1.1*y.max()
    plt.ylim([0,yciel])
    plt.vlines(conf95, 0, yciel ,colors='r')
    plt.title('Dark Matter Signal Posterior PDF (all bins)',fontsize = 16)
    plt.xlabel(r'Signal Strength [$\eta$]',fontsize = 14)
    plt.ylabel(r'Probability Density', fontsize = 14)
    plt.text(1.05*conf95,0.75*yciel, '95% conf: {0:.3f}'.format(conf95), bbox=dict(facecolor='red',alpha=0.5))
    plt.show()

# Posterior for one bin
## Takes bin number, number of walkers, number of links to burn, and number of links
def run_MCMC_onebin(binnum, nwalkers, burnlinks, links, hist_bincnt):
    ndim = 3
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda param: log_likelihood_onebin(binnum-1,param))
    progbar.set_totalsteps(float(burnlinks)*nwalkers)
    print 'Begin burn in'
    pos, prob, state = sampler.run_mcmc(p0,burnlinks)
    progbar.update_progress(1)
    print 'Burn in completed'
    sampler.reset()
    progbar.set_totalsteps(float(links)*nwalkers)
    sampler.run_mcmc(pos, links)
    progbar.update_progress(1)
    print('Main chain completed')
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    # Determine the 95% confidence level by sorting the marginalized chain and finding the 
    # eta element that is 95% of the way down the sorted chain
    sorted_eta = np.sort(sampler.flatchain[:,0])
    conf95_index = np.round(0.95*len(sorted_eta) - 1).astype(np.int)
    conf998_index = np.round(0.998*len(sorted_eta) - 1).astype(np.int)
    conf95 = sorted_eta[conf95_index]
    conf998 = sorted_eta[conf998_index]
    print '95% upper limit: ' + str(conf95)
    plt.figure()
    y, x, o = plt.hist(sampler.flatchain[:,0], hist_bincnt, range=[0,conf998], histtype="step", normed = 1)
    yciel = 1.1*y.max()
    plt.ylim([0,yciel])
    plt.vlines(conf95, 0, yciel ,colors='r')
    plt.title('Dark Matter Signal Posterior PDF (bin %d)' % binnum,fontsize = 16)
    plt.xlabel(r'Signal Strength [$\eta$]',fontsize = 14)
    plt.ylabel(r'Probability Density', fontsize = 14)
    plt.text(1.05*conf95,0.75*yciel, '95% conf: {0:.3f}'.format(conf95), bbox=dict(facecolor='red',alpha=0.5))
    plt.show()
#################################################
#   Test Code For Checking                      #
#   (Not polished, not generalized)             #
#################################################

def plot_jeffreys_prior():
    read_data("DMm400AVu_cf.txt")
    eta = np.linspace(0.1,15,200)
    out = np.zeros_like(eta)
    s = N_sig
    b = E_sig
    print s
    print b
    for i in range(0,len(eta)):
        out[i] = jeffreys((eta[i],s,b))
    plt.plot(eta,out)
    plt.show()

def test_log_likelihood():
    read_data("DMm400AVu_cf.txt")
    set_prior(uniform)
    print log_likelihood_all((0.2,0.4,0.6))
    #eta = np.linspace(0.1,0,0)
    #out = np.zeros_like(eta)
    #for i in range(0,len(eta)):
    #    out[i] = log_likelihood_all((eta[i],0,0))
    #plt.plot(eta,out)
    #plt.show()

def test_MCMC():
    read_data("DMm400AVu_cf.txt")
    set_prior(uniform)
    ndim = 3
    nwalkers = 90
    burnlinks = 300
    links = 100
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_all)
    progbar.set_totalsteps(float(burnlinks)*nwalkers)
    print 'Begin burn in'
    pos, prob, state = sampler.run_mcmc(p0,burnlinks)
    progbar.update_progress(1)
    print 'Burn in completed'
    sampler.reset()
    progbar.set_totalsteps(float(links)*nwalkers)
    sampler.run_mcmc(pos, links)
    progbar.update_progress(1)
    print('Main chain completed')
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    sorted_eta = np.sort(sampler.flatchain[:,0])
    conf95_index = np.round(0.95*len(sorted_eta) - 1).astype(np.int)
    conf998_index = np.round(0.998*len(sorted_eta) - 1).astype(np.int)
    conf95 = sorted_eta[conf95_index]
    conf998 = sorted_eta[conf998_index]
    print '95% upper limit: ' + str(conf95)
    plt.figure()
    y, x, o = plt.hist(sampler.flatchain[:,0], 300, range=[0,conf998], histtype="step", normed = 1)
    yciel = 1.1*y.max()
    plt.ylim([0,yciel])
    plt.vlines(conf95, 0, yciel ,colors='r')
    plt.title('Dark Matter Signal Posterior PDF (all bins)',fontsize = 16)
    plt.xlabel(r'Signal Strength [$\eta$]',fontsize = 14)
    plt.ylabel(r'Probability Density', fontsize = 14)
    plt.text(1.05*conf95,0.75*yciel, '95% conf: {0:.3f}'.format(conf95), bbox=dict(facecolor='red',alpha=0.5))
    plt.show()


#####################################################################################
##############################  Call the main method   ##############################
#####################################################################################
main()
