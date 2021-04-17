    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        
        return 0
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        sigma = self.sigma
        vov = self.vov
        rho = self.rho
        nstep = 600
        npath = 10000
        dt = texp / nstep
        cor = np.array([[1,rho],[rho,1]])
        cov = cor 
        chol = np.linalg.cholesky(cov)
        
        log_St = np.log(spot) * np.ones((nstep+1, npath))
        
        vol = sigma * np.ones((nstep+1, npath))
        strike_m = strike[:,None] * np.ones((strike.size, npath))
        
        
        for k in range(0, nstep):
            z = np.random.normal(loc=0,scale=1,size=[2, npath]) 
            z_corr = chol @ z 

            log_St[k + 1, :] = log_St[k, :] + vol[k,:] * np.sqrt(dt) * z_corr[0,:] - 1/2 * dt * vol[k,:]**2 
            vol[k+1, :] = vol[k, :] * np.exp(vov * np.sqrt(dt) * z_corr[1,:]-1/2 * (vov**2)*dt)
        
        St = np.exp(log_St)
        price = np.mean( np.fmax(cp*(St[-1,:] - strike_m), 0),axis=1 )
        sim_var = np.var( np.fmax(cp*(St[-1,:] - strike_m), 0),axis=1 )
        self.sim_var = sim_var
        return price

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        return 0
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        rho = self.rho
        vov = self.vov
        sigma = self.sigma
        nstep = 600
        npath = 10000
        dt = texp / nstep
        cor = np.array([[1,rho],[rho,1]])
        cov = cor
        chol = np.linalg.cholesky(cov)
        St = spot * np.ones((nstep+1, npath))
        vol = sigma * np.ones((nstep+1, npath))
        strike_m = strike[:,None] * np.ones((strike.size, npath))

        for k in range(0, nstep):
            z = np.random.normal(loc=0,scale=1,size=[2, npath]) 
            z_corr = chol @ z 
            St[k + 1, :] = St[k, :] + vol[k,:] * np.sqrt(dt) * z_corr[0,:]  
            vol[k+1, :] = vol[k, :] * np.exp(vov * np.sqrt(dt) * z_corr[1,:] - 0.5 * (vov**2)*dt)
        
        price = np.mean( np.fmax(cp*(St[-1,:] - strike_m), 0), axis=1 )
        sim_var = np.var( np.fmax(cp*(St[-1,:] - strike_m), 0), axis=1 )
        self.sim_var = sim_var
        return price

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''        
        nstep = 100
        npath = 10000
        dt = texp/nstep
        sigma = self.sigma
        vov = self.vov
        rho = self.rho
        
        vol = sigma * np.ones([nstep+1, npath])
        for i in range(nstep):
            z = np.random.randn(npath)
            vol[i+1,:] = vol[i,:] * np.exp(vov * np.sqrt(dt) * z-1/2 * (vov**2)*dt)
        
        var = vol ** 2 / sigma**2
        IT = var.mean(axis=0)
        
        spot_BS = spot * np.exp(rho*(vol[-1,:]-vol[0,:]) - 0.5* (rho*sigma)**2 *texp*IT)
        vol_BS = sigma * np.sqrt((1-rho**2)*IT)
        
        price = []
        sim_var = []
        for k in strike:
            price_onpaths = bsm.price(k, spot_BS, texp, vol_BS)
            sim_var.append(price_onpaths.var())
            price.append(price_onpaths.mean())
                
        self.sim_var = sim_var
        return price

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return 0
        
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
                
        nstep = 100
        npath = 10000
        dt = texp/nstep
        sigma = self.sigma
        vov = self.vov
        rho = self.rho
        
        vol = sigma * np.ones([nstep+1, npath])
        for i in range(nstep):
            z = np.random.randn(npath)
            vol[i+1,:] = vol[i,:] * np.exp(vov * np.sqrt(dt) * z-1/2 * (vov**2)*dt)
        
        var = vol ** 2 / sigma**2
        IT = var.mean(axis=0)
        
        spot_N = spot + rho/vov * (vol[-1,:]-vol[0,:])
        vol_N = sigma * np.sqrt((1-rho**2)*IT)
        
        price = []
        sim_var = []
        for k in strike:
            price_onpaths = normal.price(k, spot_N, texp, vol_N)
            sim_var.append(price_onpaths.var())
            price.append(price_onpaths.mean())
             
        self.sim_var = sim_var
        return price