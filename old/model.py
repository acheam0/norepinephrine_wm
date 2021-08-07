'''
Peter Duggins, Terry Stewart, Xuan Choo, Chris Eliasmith
Effects of Guanfacine and Phenylephrine on a Spiking Neuron Model of Working Memory
June-August 2016
Main Model File
'''

def run(params):
    import nengo
    from nengo.rc import rc
    import numpy as np
    import pandas as pd
    from helper import reset_gain_bias, primary_dataframe, firing_dataframe

    decision_type=params[0]
    drug_type=params[1]
    drug = params[2]
    trial = params[3]
    seed = params[4]
    P = params[5]
    dt=P['dt']
    dt_sample=P['dt_sample']
    t_cue=P['t_cue']
    t_delay=P['t_delay']
    drug_effect_neural=P['drug_effect_neural']
    drug_effect_functional=P['drug_effect_functional']
    drug_effect_biophysical=P['drug_effect_biophysical']
    enc_min_cutoff=P['enc_min_cutoff']
    enc_max_cutoff=P['enc_max_cutoff']
    sigma_smoothing=P['sigma_smoothing']
    frac=P['frac']
    neurons_inputs=P['neurons_inputs']
    neurons_wm=P['neurons_wm']
    neurons_decide=P['neurons_decide']
    time_scale=P['time_scale']
    cue_scale=P['cue_scale']
    tau=P['tau']
    tau_wm=P['tau_wm']
    noise_wm=P['noise_wm']
    noise_decision=P['noise_decision']
    perceived=P['perceived']
    cues=P['cues']

    if drug_type == 'biophysical': rc.set("decoder_cache", "enabled", "False") #don't try to remember old decoders
    else: rc.set("decoder_cache", "enabled", "True")

    def cue_function(t):
        if t < t_cue and perceived[trial]!=0:
            return cue_scale * cues[trial]
        else: return 0

    def time_function(t):
        if t > t_cue:
            return time_scale
        else: return 0

    def noise_bias_function(t):
        import numpy as np
        if drug_type=='neural':
            return np.random.normal(drug_effect_neural[drug],noise_wm)
        else:
            return np.random.normal(0.0,noise_wm)

    def noise_decision_function(t):
        import numpy as np
        if decision_type == 'default':
            return np.random.normal(0.0,noise_decision)
        elif decision_type == 'basal_ganglia':
            return np.random.normal(0.0,noise_decision,size=2)

    def inputs_function(x):
        return x * tau_wm

    def wm_recurrent_function(x):
        if drug_type == 'functional':
            return x * drug_effect_functional[drug]
        else:
            return x

    def decision_function(x):
        output=0.0
        if decision_type=='default':
            value=x[0]+x[1]
            if value > 0.0: output = 1.0
            elif value < 0.0: output = -1.0
        elif decision_type=='basal_ganglia':
            if x[0] > x[1]: output = 1.0
            elif x[0] < x[1]: output = -1.0
        return output

    def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
        pos_x = 0.5 * (x + 1)
        rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
        return rescaled

    '''model definition'''
    with nengo.Network(seed=seed+trial) as model:

        #Ensembles
        cue = nengo.Node(output=cue_function)
        time = nengo.Node(output=time_function)
        inputs = nengo.Ensemble(neurons_inputs,2)
        noise_wm_node = nengo.Node(output=noise_bias_function)
        noise_decision_node = nengo.Node(output=noise_decision_function)
        wm = nengo.Ensemble(neurons_wm,2)
        decision = nengo.Ensemble(neurons_decide,2)
        output = nengo.Ensemble(neurons_decide,1)

        #Connections
        nengo.Connection(cue,inputs[0],synapse=None)
        nengo.Connection(time,inputs[1],synapse=None)
        nengo.Connection(inputs,wm,synapse=tau_wm,function=inputs_function)
        wm_recurrent=nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
        nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
        wm_to_decision=nengo.Connection(wm[0],decision[0],synapse=tau)
        nengo.Connection(noise_decision_node,decision[1],synapse=None)
        nengo.Connection(decision,output,function=decision_function)

        #Probes
        probe_wm=nengo.Probe(wm[0],synapse=0.01,sample_every=dt_sample)
        probe_spikes=nengo.Probe(wm.neurons, 'spikes', sample_every=dt_sample)
        probe_output=nengo.Probe(output,synapse=None,sample_every=dt_sample)




    '''SIMULATION'''
    print 'Running drug \"%s\", trial %s...' %(drug,trial+1)
    with nengo.Simulator(model,dt=dt) as sim:
        if drug_type == 'biophysical': sim=reset_gain_bias(
                P,model,sim,wm,wm_recurrent,wm_to_decision,drug)
        sim.run(t_cue+t_delay)
        df_primary=primary_dataframe(P,sim,drug,trial,probe_wm,probe_output)
        df_firing=firing_dataframe(P,sim,drug,trial,sim.data[wm],probe_spikes)
    return [df_primary, df_firing]



'''MAIN'''
def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from helper import make_cues, empirical_dataframe
    from pathos.helpers import freeze_support #for Windows
    # import ipdb

    '''Import Parameters from File'''
    P=eval(open('parameters.txt').read()) #parameter dictionary
    seed=P['seed'] #sets tuning curves equal to control before drug application
    n_trials=P['n_trials']
    drug_type=str(P['drug_type'])
    decision_type=str(P['decision_type'])
    drugs=P['drugs']
    trials, perceived, cues = make_cues(P)
    P['timesteps']=np.arange(0,int((P['t_cue']+P['t_delay'])/P['dt_sample']))
    P['cues']=cues
    P['perceived']=perceived

    '''Multiprocessing'''
    print "drug_type=%s, decision_type=%s, trials=%s..." %(drug_type,decision_type,n_trials)
    freeze_support()
    exp_params=[]
    for drug in drugs:
        for trial in trials:
            exp_params.append([decision_type, drug_type, drug, trial, seed, P])
    df_list=[run(exp_params[0]),run(exp_params[-1])]
    primary_dataframe = pd.concat([df_list[i][0] for i in range(len(df_list))], ignore_index=True)
    firing_dataframe = pd.concat([df_list[i][1] for i in range(len(df_list))], ignore_index=True)

    '''Plot and Export'''
    print 'Exporting Data...'
    primary_dataframe.to_pickle('primary_data.pkl')
    firing_dataframe.to_pickle('firing_data.pkl')
    param_df=pd.DataFrame([P])
    param_df.reset_index().to_json('params.json',orient='records')

    print 'Plotting...'
    emp_dataframe=empirical_dataframe()
    sns.set(context='poster')
    figure, (ax1, ax2) = plt.subplots(2, 1)
    sns.tsplot(time="time",value="wm",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
    sns.tsplot(time="time",value="correct",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
    sns.tsplot(time="time",value="accuracy",data=emp_dataframe,unit='trial',condition='drug',
                interpolate=False,ax=ax2)
    sns.tsplot(time="time",value="accuracy",data=emp_dataframe, unit='trial',condition='drug',
                interpolate=True,ax=ax2, legend=False)
    ax1.set(xlabel='',ylabel='decoded $\hat{cue}$',xlim=(0,9.5),ylim=(0,1),
                title="drug_type=%s, decision_type=%s, trials=%s" %(drug_type,decision_type,n_trials))
    ax2.set(xlabel='time (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
    figure.savefig('primary_plots.png')

    figure2, (ax3, ax4) = plt.subplots(1, 2)
    if len(firing_dataframe.query("tuning=='weak'"))>0:
        sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax3,ci=95,
                data=firing_dataframe.query("tuning=='weak'").reset_index())
    if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
        sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax4,ci=95,
                data=firing_dataframe.query("tuning=='nonpreferred'").reset_index())
    ax3.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='Normalized Firing Rate',title='Preferred Direction')
    ax4.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='',title='Nonpreferred Direction')
    figure2.savefig('firing_plots.png')

    plt.show()

if __name__=='__main__':
    main()