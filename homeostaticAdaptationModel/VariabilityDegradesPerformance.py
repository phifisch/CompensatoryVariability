import numpy as np
from scipy import io as scio
import os
import pdb
from helperFuns import get_PN_Std_Bhandawat


DEBUG = True

class MBmodel():
    '''def __init__(self,*_,**kwargs):
        self.nKCs = kwargs.get('nKCs',2000)
        self.nPNs = kwrags.get('nPNs',24)
        
        try:
            # self.learningRate = kwargs.get('learningRate')
            self.learningRate = kwargs.get('learningRate',10**(-3))
            
        except KeyError as er:
            raise Exception(f'mandatory parameter "{str(er)}" omitted in MBmodel')
        return
        
        self._mbModelGen = MBmodelGenerator(self.nKCs,self.nPNs,**kwargs)
        '''
    def __init__(self, modelGen):
        self._mbModelGen = modelGen
        self.hasMBONs = False
        self.updateModel()
        return
    
    def updateModel(self):
        newParams = self._mbModelGen.get_model_parameters()
        self.__dict__.update(newParams)
        return
    
    def get_optimiser_parameters(self):
        return self._mbModelGen.optimizerParams
    
    def optimise(self, PNactivity):
        # self._mbModelGen.optimize_params_NadasCode(PNactivity)
        # self._mbModelGen.optimize_params_rewrite(PNactivity)
        self._mbModelGen.optimize_params(PNactivity)
        self.updateModel()
    
    def simulate(self, PNactivity: np.ndarray):
        a = [self.PNtoKC.T @ PNactivity[k] for k in range(PNactivity.shape[1])]
        totalExc = np.sum(a,axis=0)
        y = np.array([a[k]-self.alpha*totalExc-self.C_theta*self.KCtheta for k in range(PNactivity.shape[1])])
        y[y<0.] = 0
        return y
    
    def initiate_MBONs(self, MBONlist: list[bool] = [True,False] ):
        """MBONlist is an iterable of bool values which symbol the "aVersive" (V) or "aPproach" (P) characteristic
                defaults to a list of 2 values, 1 aVoid, 1 aProach
        """
        self.hasMBONs = True
        # inititate connectivity: all-to-all, lognormal weights
        self.KCtoMBON = self._modelGen.draw_lognormal_weights((self.nKCs,len(MBONlist)))
        # set a learning rate and possible related parameters
        raise NotImplementedError()
        return
    
    def learn_MBON_mapping(self, PNinput: np.ndarray[float], isGoodOdor: np.ndarray[bool]):
        raise NotImplementedError("didn't think I was gonna need that any time soon")
    
        
'''class MBmodelGenerator():
    def __init__(self, nKCs: int, nPNs: int,*_,randomSeed: float = None,**kwargs):
        self.nKCs = nKCs
        self.nPNs = nPNs
        self.rng = np.random.default_rng(seed=randomSeed)
        # ss = np.random.SeedSequence(randomSeed)
        # bg = np.random.PCG64(ss)
        pdb.set_trace()
        self.PNtoKC = self._generate_claw_connectivity(
                kwargs.get('Nclaws_mean',6,),
                kwargs.get('Nclaws_std',1.7)  )
        self._generate_claw_weights()
        self.KCtheta = self._generate_spike_thresholds(                
                kwargs.get('theta_mean',10.),
                kwargs.get('theta_std',10*5.6/21.5),
                kwargs.get('theta_limits',(0.01,70))  )

        self._params_optimized = False
        self.optimizerParams = {'Ctheta_init':1.,
                'APLgain_init': 0.01,
                'CL_incInhib':0.10, # 10% coding level (prop of active KCs)
                'CL_disInhib':0.10, # 10% coding level (prop of active KCs)
                'eta_C':1., # scales adjustment steps for C_theta
                'eta_alpha':0.0000001, # scales adjustment steps for ALPgain
                }
        self.C_theta = self.optimizerParams['Ctheta_init']
        self.alpha = self.optimizerParams['APLgain_init']
        return MBmodel(self)'''

class MBmodelBuilder():
    def __init__(self, nKCs: int, nPNs: int,*_,randomSeed: float = None,**kwargs):
        self.nKCs = nKCs
        self.nPNs = nPNs
        self.rng = np.random.default_rng(seed=randomSeed)
        # ss = np.random.SeedSequence(randomSeed)
        # bg = np.random.PCG64(ss)
        # self._PNtoKC_connected = False
        # self._PNtoKC_weighted = False
        # self._KCthresholds_generated = False
        self._params_optimized = False
        self.optimizerParams = {'Ctheta_init':1.,
                'APLgain_init': 0.000001,
                'CL_incInhib':0.10, # 10% coding level (prop of active KCs)
                'CL_disInhib':0.20, # 20% coding level (prop of active KCs)
                'eta_C':1., # scales adjustment steps for C_theta
                'eta_alpha':0.000001, # scales adjustment steps for ALPgain
                'epsilon-CL_incInh': 0.01,
                'maxLoops': 1000
                }
        self.C_theta = self.optimizerParams['Ctheta_init']
        self.alpha = self.optimizerParams['APLgain_init']
        self.Sigmoid_factor = 1. # as inverse of sigma (multiply instead divide)
        return
    
    def _generate_claw_connectivity(self, Nclaws_mean: float, Nclaws_std: float):
        # raise NotImplementedError()
        assert(Nclaws_mean>0 and Nclaws_std>=0)  
        self.Nclaws_std = Nclaws_std
        self.Nclaws_mean = Nclaws_mean
        #instantiate connectivity martrix
        connectMatrix = np.zeros((self.nPNs,self.nKCs))
        if self.Nclaws_std==0:
            #give every KC the desired amount of input PNs
            self.KCinputPNs = [self.rng.integers(0,self.nPNs, Nclaws_mean) 
                for j in range(self.nKCs)]
        else: #i.e. >0
            #give every KC a variable amount of input PNs, between 2 and 11
            Nclaws = self.rng.normal(self.Nclaws_mean, self.Nclaws_std, self.nKCs)
            Nclaws[Nclaws<2.] = 2.
            Nclaws[Nclaws>11.] =11.
            Nclaws = np.round(Nclaws).astype(int)
            self.KCinputPNs = [self.rng.integers(0,self.nPNs, Nclaws[j]) 
                for j in range(self.nKCs)]
        # now include these connections in the (empty) connectivity matrix
        for j,upstreamIdx in enumerate(self.KCinputPNs):
            connectMatrix[upstreamIdx,j] = 1.
        return connectMatrix
                
    def _generate_claw_weights(self):
        self.PNtoKC *= self.draw_lognormal_weights(self.PNtoKC.shape)
        return
        
    def _generate_spike_thresholds(self, theta_mean: float, theta_std: float, 
                        theta_limits: tuple[float,float] ) -> np.ndarray[float] :
        assert(theta_mean>0 and theta_std>=0)
        self.theta_std = theta_std
        self.theta_mean = theta_mean
        if self.theta_std == 0:
            return np.full((self.nPNs, self.nKCs), self.theta_mean)
        else:
            theta = self.rng.normal(self.theta_mean, self.theta_std, self.nKCs)
            theta[theta>theta_limits[1]] = theta_limits[1]
            theta[theta<theta_limits[0]] = theta_limits[0]
            return theta
        
    def draw_lognormal_weights(self, shape):
        """Draws log-normal distributed number centered around 1
        """
        # keep this function instead of np.random.Generator.lognormal for 
        # ease of comparison with Nada's matlab code
        return np.exp(-0.0507 + 0.3527*self.rng.standard_normal(shape) )
        
    def optimize_params(self, PNactivity):
        self.optimize_params_rewrite(PNactivity)
        return
    
    def optimize_params_NadasCode(self, X: np.ndarray):
        goodEnough = False
        detectDeadEnd = True
        nLoops = 0
        # Sigmoid_deriv = lambda x: np.exp(-0.1*x)/((1+np.exp(-0.1*x))**2)
        Sigmoid_factor = -1. #replace division by multiplication
        Sigmoid_deriv = lambda x: np.exp(-Sigmoid_factor*x)/((1+np.exp(-Sigmoid_factor*x))**2)
        # calculate coding level CL withou APL gain control
        APLgain = self.alpha
        C_theta = self.C_theta
        maxLoops = self.optimizerParams['maxLoops']
        if DEBUG:
            C_theta = self.optimizerParams['Ctheta_init']
            APLgain = self.optimizerParams['APLgain_init']
        theta = self.KCtheta.reshape([-1,1])
        # pdb.set_trace()
        if DEBUG:
            pdb.set_trace()
        if detectDeadEnd:
            APLgain_prev = APLgain
            C_theta_prev = C_theta
        while not goodEnough:
            nLoops +=1
            if nLoops>maxLoops:
                raise NotConvergingError(f'Optimisation did not converge after {nLoops} loops.', 
                      {'APLgain':APLgain, 'C_theta':C_theta } )
            # A = [self.PNtoKC.T @ X[:,k] for k in range(X.shape[1])]
            A = self.PNtoKC.T @ X
            # y_noInh = np.array([A[:,k]-C_theta*theta.flatten() for k in range(X.shape[1])])
            y_noInh = A - C_theta*theta #significantly faster
            # y_noInh[y_noInh<0.] = 0.
            CL_noInh = np.mean(y_noInh >0., axis=0)
            CL_noInh = np.mean(CL_noInh)
            # print(abs(CL_noInh-np.mean(y_noInh>0.))<0.0000001)
            # CL_noInh = np.mean(y_noInh>0.) #collapse two averaging ops
            
            #calculate gradients
            # dsig_dy = np.exp(0.9*y_noInh)/((1+np.exp(0.9*y_noInh))**2)
            dsig_dy = Sigmoid_deriv(y_noInh)
            dsig_dy[np.isnan(dsig_dy)] = 0. #what for, when does it occur?
            dEpsi_dtheta = -1.*(y_noInh>0.)*dsig_dy*theta
            # grad_theta = (CL_noInh-0.20)/(self.nKCs*X.shape[1])*np.sum(dEpsi_dtheta)
            grad_theta = (CL_noInh-self.optimizerParams['CL_disInhib'])*np.mean(dEpsi_dtheta) # rewrite simpler
            C_theta -= self.optimizerParams['eta_C']*grad_theta
            if DEBUG:
                pass
                # print(grad_theta)
            if C_theta<0:
                raise InvalidOptimisedValueError('the scale factor in the random model is negative')
            
            # optimise APLgain, recalculate after updating C_theta
            # A = [self.PNtoKC.T @ X[:,k] for k in range(X.shape[1])]
            # A = self.PNtoKC.T @ X
            totalExc = np.sum(A,axis=0) #total excitation (separate for each odor)
            # y_incInh = np.array([A[:,k]-APLgain*totalExc[k]-C_theta*theta.flatten() for k in range(X.shape[1])])
            y_incInh = A - APLgain*totalExc - C_theta*theta #significantly faster
            y_incInh[y_incInh<0.] = 0.
            CL_incInh = np.mean(y_incInh>0.)
            
            #calculate gradients
            # dsig_dy = np.exp(0.9*y_incInh)/((1+np.exp(0.9*y_incInh))**2)
            dsig_dy = Sigmoid_deriv(y_incInh)
            dsig_dy[np.isnan(dsig_dy)] = 0. #what for, when does it occur?
            dAct_dalpha = totalExc
            dsig_dalpha = -1.*(y_incInh>0.)*dAct_dalpha*dsig_dy
            # pdb.set_trace()
            grad_alpha = (CL_incInh-self.optimizerParams['CL_incInhib'])/(self.nKCs*X.shape[1])*np.sum(dsig_dalpha)
            APLgain -= self.optimizerParams['eta_alpha']*grad_alpha
            if DEBUG:
                pass
                # print(grad_alpha)
            
            # check if anothing has changed, that mean we struck a dead end
            if detectDeadEnd:
                if APLgain==APLgain_prev and C_theta==C_theta_prev:
                    raise NotConvergingError('Values remained unchanged without fulfilling the criteria',
                                {'APLgain':APLgain, 'C_theta':C_theta} )
            #check if constraints are met
            #  CL without inhibition
            # A = [self.PNtoKC.T @ X[:,k] for k in range(X.shape[1])]
            # y_noInh = np.array([A[:,k]-C_theta*theta.flatten() for k in range(X.shape[1])])
            y_noInh = A - C_theta*theta #significantly faster            y_noInh[y_noInh<0.] = 0.
            CL_noInh = np.mean(y_noInh>0.)
            #  CL including inhibition
            # A = self.PNtoKC.T @ X
            totalExc = np.sum(A,axis=0)
            y_incInh = y_noInh - APLgain*totalExc #reuse calculation
            # if DEBUG:
                # pdb.set_trace()
            # y_incInh[y_incInh<0.] = 0
            CL_incInh = np.mean(y_incInh>0.)
            
            # constraint
            if DEBUG:
                pass
                # print(nLoops, CL_noInh, CL_incInh, C_theta, APLgain)#, grad_theta, grad_alpha)
                # print(grad_theta), grad_alpha)
                # pdb.set_trace()
            goodEnough = (np.abs(CL_noInh/CL_incInh-2.0) <0.2) and (np.abs(CL_incInh-self.optimizerParams['CL_incInhib'])<self.optimizerParams['epsilon-CL_incInh'])
        
        print(f'Optimisation took {nLoops} loops')
        #now set the parameters in odel
        self._params_optimized = True
        self.C_theta = C_theta
        self.alpha = APLgain
        return
    
    def _adjust_C_theta(self, A, APLgain, C_theta, theta): #APLgain not used, but keep in case derived classes need it
        y_noInh = A - C_theta*theta #significantly faster
        y_noInh[y_noInh<0.] = 0. #seems useless at first, but helps to prevent getting stuck by getting more dsig_dy>>0.
        # y_noInh += 0.01*self.rng.normal(size=y_noInh.shape)
        CL_noInh = np.mean(y_noInh>0.) #collapse two averaging ops
        
        #calculate gradients
        # y_noInh[y_noInh<0.] = 0.
        dsig_dy = self.Sigmoid_deriv(y_noInh)
        dsig_dy[np.isnan(dsig_dy)] = 0. #what for, when does it occur?
        dEpsi_dtheta = -1.*(y_noInh>0.)*dsig_dy*theta
        # grad_theta = (CL_noInh-0.20)/(self.nKCs*X.shape[1])*np.sum(dEpsi_dtheta)
        grad_theta = (CL_noInh-self.optimizerParams['CL_disInhib'])*np.mean(dEpsi_dtheta) # rewrite simpler
        C_theta -= self.optimizerParams['eta_C'] * grad_theta
        if DEBUG:
            pass
            # print(grad_theta)
        return C_theta

    def _adjust_alpha(self, A, APLgain, C_theta, theta):
        totalExc = np.sum(A,axis=0) #total excitation (separate for each odor)
        y_incInh = A - APLgain*totalExc - C_theta*theta #significantly faster
        # y_incInh += 0.01*self.rng.normal(size=y_incInh.shape)
        # y_incInh[y_incInh<0.] = 0.
        CL_incInh = np.mean(y_incInh>0.)
        
        #calculate gradients
        # dsig_dy = np.exp(0.9*y_incInh)/((1+np.exp(0.9*y_incInh))**2)
        dsig_dy = self.Sigmoid_deriv(y_incInh)
        dsig_dy[np.isnan(dsig_dy)] = 0. #what for, when does it occur?
        dAct_dalpha = totalExc
        dsig_dalpha = -1.*(y_incInh>0.)*dAct_dalpha*dsig_dy
        grad_alpha = (CL_incInh - self.optimizerParams['CL_incInhib']) * np.mean(dsig_dalpha)
        APLgain -= self.optimizerParams['eta_alpha'] * grad_alpha
        if DEBUG:
            pass
            # print(grad_alpha)
        return APLgain
    
    def Sigmoid_deriv(self, x):
        return np.exp(-self.Sigmoid_factor*x)/((1+np.exp(-self.Sigmoid_factor*x))**2)

    def optimize_params_rewrite(self, PNactivity):
        goodEnough = False
        detectDeadEnd = True
        nLoops = 0
        maxLoops = self.optimizerParams['maxLoops']
        APLgain = self.alpha
        C_theta = self.C_theta
        theta = self.KCtheta.reshape([-1,1])
        # pdb.set_trace()
        if DEBUG:
            pdb.set_trace()
        if DEBUG:
            C_theta = self.optimizerParams['Ctheta_init']
            APLgain = self.optimizerParams['APLgain_init']
        if detectDeadEnd:
            APLgain_prev = APLgain
            C_prev = C_theta
        while not goodEnough:
            nLoops +=1
            if nLoops>maxLoops:
                raise NotConvergingError(f'Optimisation did not converge after {nLoops} loops.', 
                      {'APLgain':APLgain, 'C_theta':C_theta } )
            A = self.PNtoKC.T @ PNactivity
            
            C_theta = self._adjust_C_theta(A, APLgain, C_theta, theta)
            if C_theta<0:
                raise InvalidOptimisedValueError('the scale factor in the random model is negative')

            # optimise APLgain, recalculate after updating C_theta
            APLgain = self._adjust_alpha(A, APLgain, C_theta, theta)
            
            # check if anothing has changed, that means we struck a dead end
            if detectDeadEnd:
                if APLgain==APLgain_prev and C_theta==C_prev:
                    raise NotConvergingError('Values remained unchanged without fulfilling the criteria',
                                {'APLgain':APLgain, 'C_theta':C_theta } )
                APLgain_prev = APLgain
                C_prev = C_theta
            #check if constraints are met
            #  CL without inhibition
            y_noInh = A - C_theta*theta
            CL_noInh = np.mean(y_noInh>0.)
            #  CL including inhibition
            totalExc = np.sum(A,axis=0)
            y_incInh = y_noInh - APLgain*totalExc #reuse calculation
            CL_incInh = np.mean(y_incInh>0.)
            
            # constraint
            if DEBUG:
                pass
                # print(nLoops, CL_noInh, CL_incInh, C_theta, APLgain)
                # pdb.set_trace()
            goodEnough = (np.abs(CL_noInh/CL_incInh-2.0) <0.2) and (np.abs(CL_incInh-self.optimizerParams['CL_incInhib']) 
                             <self.optimizerParams['epsilon-CL_incInh'] )
        
        print(f'Optimisation took {nLoops} loops')
        if APLgain<0:
            raise InvalidOptimisedValueError('the APL factor in the random model is negative')
        #now set the parameters in odel
        self._params_optimized = True
        self.C_theta = C_theta
        self.alpha = APLgain
        return
    
    def optimize_params_different_operations_order(self, PNactivity):
        goodEnough = False
        detectDeadEnd = True
        nLoops = 0
        #replace division by multiplication in sigmoid calculation
        Sigmoid_factor = 1/0.1 
        Sigmoid_deriv = lambda x: np.exp(-Sigmoid_factor*x)/((1+np.exp(-Sigmoid_factor*x))**2)
        # calculate coding level CL withou APL gain control
        APLgain = self.alpha
        C_theta = self.C_theta
        if DEBUG:
            C_theta = self.optimizerParams['Ctheta_init']
            APLgain = self.optimizerParams['APLgain_init']
        theta = self.KCtheta        
        if DEBUG:
            pdb.set_trace()
        if detectDeadEnd:
            APLgain_prev = APLgain
            C_theta_prev = C_theta
        while True:
            nLoops +=1
            if nLoops>1000:
                raise Exception('Optimisation did not converge')
            # calculate coding level with disinhibited KCs (APL gain = 0)
            A = self.PNtoKC.T @ X
            y_noInh = A - C_theta*theta
            # y_noInh[y_noInh<0.] = 0.
            # assert(CL_noInh.size == PNactivity.shape[1])#for debugging
            CL_noInh = np.mean(y_noInh>0.) #collapse two averaging ops
            
            #including APL gain control, reuse the "A" variable
            totalExc = np.sum(A,axis=0)
            y_incInh = y_noInh - APLgain*totalExc
            CL_incInh = np.mean(y_incInh>0.)
            
            #test if we are close enough
            goodEnough = ( (np.abs(CL_noInh/CL_incInh-2.0) <0.2) and 
                    (np.abs(CL_incInh-self.optimizerParams['CL_incInhib'])<0.01) )
            if goodEnough:
                break

            #calculate gradient for theta
            dSig_dy = Sigmoid_deriv(y_noInh)
            # dSig_dy[np.isnan(dSig_dy)] = 0. #what for, when does it occur?
            dEpsi_dtheta = -1.*(y_noInh>0.)*dSig_dy*theta
            # grad = (CL_noInh-0.20)/(self.nKCs*PNactivity.shape[1])*np.sum(dEpsi_dtheta)
            grad = (CL_noInh-self.optimizerParams['CL_disInhib'])*np.mean(dEpsi_dtheta) # rewrite simpler
            C_theta -= self.optimizerParams['eta_C']*grad
            
            #calculate gradient for APL gain
            dSig_dy = Sigmoid_deriv(y_incInh)
            # dSig_dy[np.isnan(dSig_dy)] = 0. #what for, when does it occur?
            dEpsi_dalpha = -1.*(y_incInh>0.)*dSig_dy*totalExc
            # grad = (CL_noInh-0.20)/(self.nKCs*PNactivity.shape[1])*np.sum(dEpsi_dtheta)
            grad_alpha = (CL_incInh-self.optimizerParams['CL_incInhib'])*np.mean(dEpsi_dalpha) # rewrite simpler
            APLgain -= self.optimizerParams['eta_alpha']*grad_alpha
             
            # check if anothing has changed, that mean we struck a dead end
            if detectDeadEnd:
                if APLgain==APLgain_prev and C_theta==C_theta_prev:
                    raise Exception('Values remained unchanged without fulfilling the criteria!')
       
        #now set the parameters in model
        self._params_optimized = True
        self.C_theta = C_theta
        self.alpha = APLgain
        return

    def get_connectivity_matrix(self):
        return self.PNtoKC
    def get_spike_thresholds(self):
        return self.KCtheta
    def get_nPNs(self):
        return self.nPNs
    def get_nKCs(self):
        return self.nKCs
    def get_model_parameters(self):
        d = {'nPNs':self.nPNs,
                'nKCs':self.nKCs,
                'alpha': self.alpha, # APLgain 
                'KCtheta':self.KCtheta, #spike_thresholds
                'C_theta':self.C_theta, # constant factor for KCtheta
                'PNtoKC': self.PNtoKC #PN to KC weights
                } 
        # update this general dict with optimised parameters to simplify adding new parameters
        return d | self.get_optimised_parameters() # optimised should include model-specific
    
    def build(self,**kwargs):
        self.PNtoKC = self._generate_claw_connectivity(
                kwargs.get('Nclaws_mean',6,),
                kwargs.get('Nclaws_std',1.7)  )
        self._generate_claw_weights()
        self.KCtheta = self._generate_spike_thresholds(                
                kwargs.get('theta_mean',10.),
                kwargs.get('theta_std',10*5.6/21.5),
                kwargs.get('theta_limits',(0.01,70))  )
        return MBmodel(self)
        
    def get_optimised_parameters(self):
        return {'C_theta': self.C_theta, 'alpha': self.alpha }
        

class MBmodelBuilder_homeostaticAbstractClass(MBmodelBuilder):
    """This class only adds certain parameters in its constructor.
    Like the name says, this is an abstract class meant for homeostatic 
    models to inherit from."""
    def __init__(self,*args,**kwargs):
        super(MBmodelBuilder_homeostaticAbstractClass, self).__init__(*args,**kwargs)
        self.optimizerParams['lifetime-sparseness-A0'] = 0.51
        self.optimizerParams['epsilon_A0'] = 0.06*self.optimizerParams['lifetime-sparseness-A0']
        self.optimizerParams['maxLoops'] = 10000
        return


class MBmodelBuilder_homeostaticThreshold(MBmodelBuilder_homeostaticAbstractClass):
    """aka the magenta model"""
    def __init__(self,*args,**kwargs):
        super(MBmodelBuilder_homeostaticThreshold, self).__init__(*args,**kwargs)
        # define extra parameters, taken from Nada's code
        self.optimizerParams['eta_theta'] = 0.1 # originally 0.01
        return
        
    def _adjust_theta(self, A, APLgain, C_theta, theta):
        y_incInh = A - APLgain*np.sum(A,axis=0) - C_theta*theta
        y_incInh[y_incInh<0.] = 0. #yes this time I need it
        avgAKcs = np.mean(y_incInh,axis=1) #lifetime average activity (mean actoss trials)
        # am we sure that we shouldn't take y_incInh>0, because A0 is only 0.51
        # the present way is directly taken from Nada's code
        errorInActivity = avgAKcs - self.optimizerParams['lifetime-sparseness-A0']
        errorInActivity = errorInActivity.reshape(theta.shape)
        theta += self.optimizerParams['eta_theta'] * C_theta * errorInActivity #yes += (-*-)
        theta[theta<0.] = 0.;
        return theta
    
    def optimize_params(self, PNactivity):
        goodEnough = False
        detectDeadEnd = True
        nLoops = 0
        maxLoops = self.optimizerParams['maxLoops']
        APLgain = self.alpha
        C_theta = self.C_theta
        theta = self.KCtheta.reshape([-1,1])
        # pdb.set_trace()
        if DEBUG:
            pdb.set_trace()
        if DEBUG:
            C_theta = self.optimizerParams['Ctheta_init']
            APLgain = self.optimizerParams['APLgain_init']
        if detectDeadEnd:
            APLgain_prev = APLgain
            C_prev = C_theta
            theta_prev = theta
        while not goodEnough:
            nLoops +=1
            if nLoops>maxLoops:
                raise NotConvergingError(f'Optimisation did not converge after {nLoops} loops.', 
                      {'APLgain':APLgain, 'C_theta':C_theta, 'KCtheta':theta } )
            A = self.PNtoKC.T @ PNactivity
            
            C_theta = self._adjust_C_theta(A, APLgain, C_theta, theta)
            if C_theta<0:
                raise InvalidOptimisedValueError('the scale factor in the random model is negative')
            
            # optimise APLgain, recalculate after updating C_theta
            APLgain = self._adjust_alpha(A, APLgain, C_theta, theta)

            theta = self._adjust_theta(A, APLgain, C_theta, theta)
            
            # check if nothing has changed, that mean we struck a dead end
            if detectDeadEnd:
                if APLgain==APLgain_prev and C_theta==C_prev and np.all(theta==theta_prev):
                                {'APLgain':APLgain, 'C_theta':C_theta, 'KCtheta':theta } )
                    raise NotConvergingError('Values remained unchanged without fulfilling the criteria',
                APLgain_prev, C_prev, theta_prev = APLgain, C_theta, theta
            #check if constraints are met
            #  CL without inhibition
            y_noInh = A - C_theta*theta
            CL_noInh = np.mean(y_noInh>0.)
            #  CL including inhibition
            totalExc = np.sum(A,axis=0)
            y_incInh = y_noInh - APLgain*totalExc #reuse calculation
            CL_incInh = np.mean(y_incInh>0.)
            y_incInh[y_incInh<0.] = 0.
            avgAKcs = np.mean(y_incInh, axis=1)
            
            # constraint
            if DEBUG:
                print(nLoops, CL_noInh, CL_incInh, C_theta, APLgain)
                # pdb.set_trace()
            goodEnough = ( np.all(np.abs(avgAKcs-self.optimizerParams['lifetime-sparseness-A0'])<self.optimizerParams['epsilon_A0']) and
                        (np.abs(CL_noInh/CL_incInh-2.0) <0.2) and
                        (np.abs(CL_incInh-self.optimizerParams['CL_incInhib'])<self.optimizerParams['epsilon-CL_incInh']) 
                        )
        
        print(f'Optimisation took {nLoops} loops')
        if APLgain<0:
            raise InvalidOptimisedValueError('the APL factor in the random model is negative')
        #now set the parameters in odel
        self._params_optimized = True
        self.C_theta = C_theta
        self.alpha = APLgain
        self.KCtheta = theta
        return

    def get_optimised_parameters(self):
        return {'C_theta': self.C_theta, 'alpha': self.alpha, 
                'KCtheta': self.KCtheta}

class MBmodelBuilder_homeostaticExcitation(MBmodelBuilder_homeostaticAbstractClass):
    """a.k.a. the blue model"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # define extra parameters, taken from Nada's code
        self.optimizerParams['eta_weights'] = 0.2 #originally 0.05
        pass
    
    def build(self,**kwargs):
        model = super(MBmodelBuilder_homeostaticExcitation, self).build(**kwargs)
        self.PNtoKCmask = (self.PNtoKC.T > 0.).astype(float) # offload to avoid repeated calculations
        return model
    
    def _adjust_PNtoKCweights(self, A, APLgain, C_theta, theta, PNtoKC):
        y_incInh = A - APLgain*np.sum(A,axis=0) - C_theta*theta
        y_incInh[y_incInh<0.] = 0. #yes this time I need it
        avgAKcs = np.mean(y_incInh,axis=1) #lifetime average activity (mean actoss trials)
        errorInActivity = avgAKcs - self.optimizerParams['lifetime-sparseness-A0']
        # if KC_j was too active, decrease all post-synapses of KC_j by the same amount
        PNtoKC -= (self.optimizerParams['eta_weights'] * 
            self.PNtoKCmask *
            errorInActivity.reshape([-1,1]) ) #assert shape
        PNtoKC[PNtoKC<0.] = 0. #then prune those that make no more sense
        return PNtoKC
        
    def optimize_params(self,PNactivity):
        goodEnough = False
        detectDeadEnd = True
        nLoops = 0
        maxLoops = self.optimizerParams['maxLoops']
        APLgain = self.alpha
        C_theta = self.C_theta
        theta = self.KCtheta.reshape([-1,1])
        PNtoKC = self.PNtoKC.T #include the transposition here (!)
        # pdb.set_trace()
        if DEBUG:
            pdb.set_trace()
        if DEBUG:
            C_theta = self.optimizerParams['Ctheta_init']
            APLgain = self.optimizerParams['APLgain_init']
        if detectDeadEnd:
            APLgain_prev = APLgain
            C_prev = C_theta
            PNtoKC_prev = PNtoKC
        while not goodEnough:
            if nLoops>maxLoops:
                raise NotConvergingError(f'Optimisation did not converge after {nLoops} loops.', 
                      {'APLgain':APLgain, 'C_theta':C_theta, 'PNtoKC':PNtoKC } )
            nLoops += 1
            A = PNtoKC @ PNactivity
            
            C_theta = self._adjust_C_theta(A, APLgain, C_theta, theta)
            if C_theta<0:
                raise InvalidOptimisedValueError('the scale factor in the random model is negative')
            
            # optimise APLgain, recalculate after updating C_theta
            APLgain = self._adjust_alpha(A, APLgain, C_theta, theta)

            PNtoKC = self._adjust_PNtoKCweights(A, APLgain, C_theta, theta, PNtoKC)
            
            if DEBUG:
                print(nLoops)
 
            # check if nothing has changed, that mean we struck a dead end
            if detectDeadEnd:
                if APLgain==APLgain_prev and C_theta==C_prev and np.all(PNtoKC==PNtoKC_prev):
                    raise NotConvergingError('Values remained unchanged without fulfilling the criteria',
                                {'APLgain':APLgain, 'C_theta':C_theta, 'PNtoKC':PNtoKC } )
                APLgain_prev, C_prev, PNtoKC_prev = APLgain, C_theta, PNtoKC
            # check if conditions are met
            goodEnough = self._check_constraints(A, APLgain, C_theta, theta, PNtoKC)
        
        print(f'Optimisation took {nLoops} loops')
        if APLgain<0:
            raise InvalidOptimisedValueError('the APL factor in the random model is negative')
        #now set the parameters in odel
        self._params_optimized = True
        self.C_theta = C_theta
        self.alpha = APLgain
        self.PNtoKC = PNtoKC.T #undo the transpose iff it was done before loop
        return
    
    def _check_constraints(self, A, APLgain, C_theta, theta, PNtoKC) ->bool :
        # coding level without inhibition
        y_noInh = A - C_theta*theta
        CL_noInh = np.mean(y_noInh>0.)
        # coding level including inhibition
        y_incInh = y_noInh - APLgain* np.sum(A,axis=0)
        CL_incInh = np.mean(y_incInh>0.)
        # lifetime sparseness (average responses of KCj to all odors)
        y_incInh[y_incInh<0.] = 0.
        avgAKcs = np.mean(y_incInh, axis=1)
        if DEBUG:
            print(CL_noInh, CL_incInh, C_theta, APLgain)
            # pdb.set_trace()
        conditionsFulfilled = ( np.all(np.abs(avgAKcs-self.optimizerParams['lifetime-sparseness-A0']) 
                                      <self.optimizerParams['epsilon_A0']) and
                     ( np.abs(CL_noInh/CL_incInh-2.0) <0.2) and
                     ( np.abs(CL_incInh-self.optimizerParams['CL_incInhib']) 
                                < self.optimizerParams['epsilon-CL_incInh']) )        
        return conditionsFulfilled

    def get_optimised_parameters(self):
        return {'C_theta': self.C_theta, 'alpha': self.alpha, 'PNtoKC': self.PNtoKC}



class NotConvergingError(Exception):
    def __init__(self, message, currentState: dict = {}, *args,**kwargs):
        super().__init__(message)
        if currentState: #is not empty
            self._write_log(currentState, *args,**kwargs)
            self.currentState = currentState
        
    def _write_log(self, currentState: dict, filename: str = None):
        if not filename:
            filename = 'errorReport_stateOfOptimisation_parameter.txt'
        with open(filename, 'w') as fl:
            for key, value in currentState.items():
                fl.write(str(key))
                fl.write('\n'+ '-'*21 + '\n')
                fl.write(str(value))
                fl.write('\n'+ '-'*42 + '\n')
        return     

class InvalidOptimisedValueError(Exception):
    pass

class OdorResponses():
    def __init__(self,*_,**kwargs):
        self.odorResp = kwargs.get('odorResponses',None)
        self.odorResp = np.array(self.odorResp)
        if self.odorResp is not None:
            self.nPNs = kwargs.get('nPNs',self.odorResp.shape[0])
        else:
            self.nPNs = kwargs.get('nPNs',None)
            
    def add_odor_resp(self, resp_matrix):
        """add odor responses as a matrix with nPN rows and K columns (K number od odor)"""
        if self.odorResp is not None:
            self.odorResp = np.hstack((self.odorResp,resp_matrix))
        else:
            if self.nPNs is not None:
                assert(len(resp_matrix)==self.nPNs)
            self.odorResp = np.array(resp_matrix)
        return
    def get_odor_resp(self):
        return self.odorResp
    def get_noisy_trials(self,noiseAmplitude, baseSignal, nTrials=1,*_,**kwargs):
        """adds gaussian noise with provided amplitude to the data
        maybe extend class/method to provide and return a seed
        arguments:
            noiseAmplitude
            baseSignal: what signal to add the noise to
            nTrials: how many noisy trials to generate, default 1
        extra keyword arguments:
            randomSeed: float, which will lead to a 
        reset of the random generator if present
        """
        if 'randomSeed' in kwargs:
            self._rng = np.random.default_rng(seed = kwargs['randomSeed'])
        elif not '_rng' in self.__dict__:
            self._rng = np.random.default_rng()
        noise = noiseAmplitude[...,np.newaxis] * self._rng.standard_normal((*baseSignal.shape,nTrials))
        # noise = noise.reshape(*baseSignal.shape,nTrials)
        return baseSignal[...,np.newaxis] + noise
    
    def __getitem__(self,key):
        return self.odorResp[key]
    
        

if __name__=='__main__':
    odorSetSize = 200
    nPNs = 24
    # loading Hallem-Carlson data
    halol_respData = scio.loadmat('hallem_olsen.mat')
    halol_respData = halol_respData['hallem_olsen']
    halol_respData = halol_respData[:110,:]
    odores = OdorResponses(odorResponses=halol_respData)

    # create artificial odors, n odors
    allActivityBins = np.empty((0,101)) #container
    numberGeneratedOdors = odorSetSize-100
    generatedOdorResponses = np.empty((nPNs, numberGeneratedOdors))
    ranGen = np.random.default_rng()
    for pn in range(nPNs):
        [activityProb,activityBins] = np.histogram(halol_respData[pn,:], 100) #drop-in for matlab histc
        activityProb = activityProb/activityProb.sum()
        allActivityBins = np.vstack((allActivityBins ,activityBins))
        generatedOdorResponses[pn,:] = ranGen.choice(
            (activityBins[:-1]+activityBins[1:])/2, numberGeneratedOdors, replace=True, p=activityProb )
    # add these new odors to our set
    #odores.add_odor_resp(generatedOdorResponses)

    # additional set of odor responses
    otherOdorResponses = scio.loadmat(os.path.abspath('../Data_submitted_fly_wNoise11.mat'))
    otherOdorResponses = otherOdorResponses['PNtrials']
    otherOdorResponses = otherOdorResponses[:,:,0]
    #odores.add_odor_resp(otherOdorResponses)


    ## recover the rescaling factors
    # get the maximum bin center for each PN derived from the original Hallem-Olsen data
    maxRespBinPerPN = np.max(allActivityBins,axis=1)
    # get the maximum response in the rescaled randomly resampled PNs 
    maxRespPNsRescaledPerPN = np.max(otherOdorResponses,axis=1)
    PNsAboveBestFit = np.ones(nPNs,dtype=bool)
    # In this while loop:
    # draw a best fit line comparing the maximum response in the rescaled
    # randomly resample PNs to the maximum bin center for each PN in the
    # original H-O data. Most PNs will match, but in some PNs, by random chance
    # they will not have sampled the top response in 100 odors. These PNs will
    # be below the best fit line, while the "matching" PNs will be above. On
    # the next iteration of the while loop, redraw the best fit using only the
    # PNs that lie above the best fit line from the current iteration
    # continue until none of the PNs are above the best fit line (because they
    # lie on it almost exactly)

    while sum(PNsAboveBestFit):
        print(sum(PNsAboveBestFit))
        p = np.polynomial.polynomial.polyfit(maxRespBinPerPN[PNsAboveBestFit], maxRespPNsRescaledPerPN[PNsAboveBestFit],1)
        # the correct PNs will be above the best fit line because the incorrect PNs
        # are outliers that drag down the line of best fit
        PNsAboveBestFit = maxRespPNsRescaledPerPN > ( maxRespBinPerPN*p[0] + p[1] +0.000001)
        # the 0.000001 is for rounding errors, otherwise you end up in endless loops
    # my own alternative idea: since we only want to get those that reach a stable max line, why not divde and take a histogram
    # -> hinges on y offset

    PNs_1 = (otherOdorResponses-p[1])/p[0]
    PNs_2 = (generatedOdorResponses*p[0])+p[1]
    allOdorResponses = OdorResponses(odorResponses=PNs_1)
    allOdorResponses.add_odor_resp(PNs_2) #now this is the 24 by K odors matrix (24 ORNs)

    # for K_odors in [50,100,150,200]:
    K_odors = 100
    nTrials = 15
    
    x = allOdorResponses[:,:K_odors]
    PNtrials = np.zeros((nPNs,K_odors,nTrials))
    PNtrials[:,:,0] = x
    noiseLevels_Bhandawat = get_PN_Std_Bhandawat(x)
    # for tr in range(1,nTrials):
        # PNtrials[:,:,tr] = allOdorResponses.get_noisy_trials(
                              # 1.*noiseLevels_Bhandawat,
                              # x, nTrials=1, randomSeed=2024  )
    PNtrials[:,:,1:] = allOdorResponses.get_noisy_trials(
                              1.*noiseLevels_Bhandawat,
                              x, nTrials=nTrials-1, randomSeed=2024  )
    PNtrials = PNtrials.reshape([PNtrials.shape[0], -1])
                            
    # add noise from Bhandawat et al.
    # PNtrials = allOdorResponses.get_noisy_trials(
    
    # % making sure that PN responses are +ve
    # %rescaling the PN responses in range from 0 to 5
    # %  --> WHY?
    # PNtrials=rescale(PNtrials,0,5);
    PNtrials[PNtrials<0.] = 0.
    PNtrials = 5*( PNtrials - PNtrials.min() 
                    )/( PNtrials.max() - PNtrials.min())
    
    thisMB = MBmodelBuilder(nKCs=2000,nPNs=24).build()
    # thisMB.optimise(PNtrials)
    # thisMB._mbModelGen.optimize_params_NadasCode(PNtrials)
    # thisMB._mbModelGen.optimize_params_rewrite(PNtrials)