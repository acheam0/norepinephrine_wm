def make_addon(N):
    import string
    import random
    addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
    return addon

def ch_dir():
    #change directory for data and plot outputs
    import os
    import sys
    root=os.getcwd()
    addon=make_addon(9)
    datadir=''
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        datadir=root+'/data/'+addon+'/' #linux or mac
    elif sys.platform == "win32":
        datadir=root+'\\data\\'+addon+'\\' #windows
    os.makedirs(datadir)
    os.chdir(datadir)
    return datadir

def empirical_dataframe():
    import numpy as np
    import pandas as pd
    columns=('time','drug','accuracy','trial')
    emp_times = [3.0,5.0,7.0,9.0]
    emp_dataframe = pd.DataFrame(columns=columns,index=np.arange(0, 12))
    pre_PHE=[0.972, 0.947, 0.913, 0.798]
    pre_GFC=[0.970, 0.942, 0.882, 0.766]
    post_GFC=[0.966, 0.928, 0.906, 0.838]
    post_PHE=[0.972, 0.938, 0.847, 0.666]
    q=0
    for t in range(len(emp_times)):
        emp_dataframe.loc[q]=[emp_times[t],'control (empirical)',np.average([pre_GFC[t],pre_PHE[t]]),0]
        emp_dataframe.loc[q+1]=[emp_times[t],'PHE (empirical)',post_PHE[t],0]
        emp_dataframe.loc[q+2]=[emp_times[t],'GFC (empirical)',post_GFC[t],0]
        q+=3
    return emp_dataframe

def make_cues(P):
    import numpy as np
    trials=np.arange(P['n_trials'])
    perceived=np.ones(P['n_trials']) #list of correctly perceived (not necessarily remembered) cues
    rng=np.random.RandomState(seed=P['seed'])
    cues=2*rng.randint(2,size=P['n_trials'])-1 #whether the cues is on the left or right
    for n in range(len(perceived)):
        if rng.rand()<P['misperceive']: perceived[n]=0
    return trials, perceived, cues


'''drug approximations'''
import nengo
class MySolver(nengo.solvers.Solver):
    #When the simulator builds the network, it looks for a solver to calculate the decoders
    #instead of the normal least-squares solver, we define our own, so that we can return
    #the old decoders
    def __init__(self,weights): #feed in old decoders
        self.weights=False #they are not weights but decoders
        self.my_weights=weights
    def __call__(self,A,Y,rng=None,E=None): #the function that gets called by the builder
        return self.my_weights.T, dict()

def reset_gain_bias(P,model,sim,wm,wm_recurrent,wm_to_decision,drug):
    #set gains and biases as a constant multiple of the old values
    wm.gain = sim.data[wm].gain * P['drug_effect_biophysical'][drug][0]
    wm.bias = sim.data[wm].bias * P['drug_effect_biophysical'][drug][1]
    #set the solver of each of the connections coming out of wm using the custom MySolver class
    #with input equal to the old decoders. We use the old decoders because we don't want the builder
    #to optimize the decoders to the new gainbias/bias values, otherwise it would "adapt" to the drug
    wm_recurrent.solver = MySolver(sim.model.params[wm_recurrent].weights)
    wm_to_decision.solver=MySolver(sim.model.params[wm_to_decision].weights)
    #rebuild the network to affect the gain/bias change
    sim=nengo.Simulator(model,dt=P['dt'])
    return sim

'''dataframe initialization'''
def primary_dataframe(P,sim,drug,trial,probe_wm,probe_output):
    import numpy as np
    import pandas as pd
    columns=('time','drug','wm','output','correct','trial')
    df_primary = pd.DataFrame(columns=columns, index=np.arange(0,len(P['timesteps'])))
    i=0
    for t in P['timesteps']:
        wm_val=np.abs(sim.data[probe_wm][t][0])
        output_val=sim.data[probe_output][t][0]
        correct=get_correct(P['cues'][trial],output_val)
        rt=t*P['dt_sample']
        df_primary.loc[i]=[rt,drug,wm_val,output_val,correct,trial]
        i+=1
    return df_primary

def firing_dataframe(P,sim,drug,trial,sim_wm,probe_spikes):
    import numpy as np
    import pandas as pd
    columns=('time','drug','neuron-trial','tuning','firing_rate')
    df_firing = pd.DataFrame(columns=columns, index=np.arange(0,len(P['timesteps'])*\
        int(P['neurons_wm']*P['frac'])))
    t_width = 0.2
    t_h = np.arange(t_width / P['dt']) * P['dt'] - t_width / 2.0
    h = np.exp(-t_h ** 2 / (2 * P['sigma_smoothing'] ** 2))
    h = h / np.linalg.norm(h, 1)
    j=0
    for nrn in range(int(P['neurons_wm']*P['frac'])):
        enc = sim_wm.encoders[nrn]
        tuning = get_tuning(P,trial,enc)
        spikes = sim.data[probe_spikes][:,nrn]
        firing_rate = np.convolve(spikes,h,mode='same')
        for t in P['timesteps']:
            rt=t*P['dt_sample']
            df_firing.loc[j]=[rt,drug,nrn+trial*P['neurons_wm'],tuning,firing_rate[t]]
            j+=1
        # print 'appending dataframe for neuron %s' %f
    return df_firing

def get_correct(cue,output_val):
    if (cue > 0.0 and output_val > 0.0) or (cue < 0.0 and output_val < 0.0): correct=1
    else: correct=0
    return correct

def get_tuning(P,trial,enc):
    cue=P['cues'][trial]
    enc_min_cutoff=P['enc_min_cutoff']
    enc_max_cutoff=P['enc_max_cutoff']
    if (cue > 0.0 and 0.0 < enc[0] < enc_min_cutoff) or \
        (cue < 0.0 and 0.0 > enc[0] > -1.0*enc_min_cutoff): tuning='superweak'
    if (cue > 0.0 and enc_min_cutoff < enc[0] < enc_max_cutoff) or \
        (cue < 0.0 and -1.0*enc_max_cutoff < enc[0] < -1.0*enc_min_cutoff): tuning='weak'
    elif (cue > 0.0 and enc[0] > enc_max_cutoff) or \
        (cue < 0.0 and enc[0] < -1.0*enc_max_cutoff): tuning='strong'
    else: tuning='nonpreferred'
    return tuning