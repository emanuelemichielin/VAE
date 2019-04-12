import numpy as np
import rqpy as rp
import qetpy as qp
import vae


__all__ = ["pre_process_PD2"]



def pre_process_PD2(path, fs, ioffset, rload, rsh, qetbias, chan, det, ds_factor=6, trunc=0,
                    lgcsave=False, savepath = ''):
    """
    Function preprocess raw traces for PD2 detector. Can be generalized for multichannel detectors. 
    
    The beginning of all traces are chopped off, set by the variable trunc. The traces are downsampled
    by a factor of ds_factor. And the traces are converted to units of power, rather than current, so 
    that they should be invairant of bias point. 
    
    Parameters
    ----------
    path : str
        Absolute path to .mid.gz dump to open
    fs : int
        Sample rate 
    ioffset : float
        The the squid offset (found either from muons or IV). Note, this number is very
        important for this preprocessing method. If unknown, this function is worthless
    rload : float
        The load resistance of the device
    rsh : float
        The value of the shunt resistor
    qetbias : float
        The applied bias current to the TES
    chan : str
        The channel name ie. 'PBS1'
    det : str
        The detector name ie. 'Z1'
    ds_factor : int, optional
        Factor to downsample the traces by
    trunc : int, optional
        The number of bins to remove from the beginning of the trace (only use if the 
        events were not triggered at the beginning of the trace)
    lgcsave : boolean, optional
        If True, the processed dumps are saved in savepath. defaults to False
    savepath : str, optional
        The path where the processed dumps should be saved. Ignored if not lgcsave
        
    Returns
    -------
    traces : dict
        Dictionary containing:
            'traces' : the preprocessed traces of shape (#traces,#channels,#bins)
            'eventnumber' : the series number concatenated with event number
    metadata : dict
        Dictionary containing:
            'scale_factor' : the sum of all traces devided by tracelength
            'n_traces' : the number of traces in the dump
            'dump' : the dump file number
            
    """
    x, info_dict = rp.io.get_traces_midgz(path, chan, det, lgcskip_empty=True, lgcreturndict=True)
    x = x[..., trunc:]
    ser_ev = np.zeros(x.shape[0], dtype=int)
    series = path.split('/')[-2]
    for ii, ev in enumerate(info_dict['eventnumber']):
        ser_ev[ii] = f'{series}_{ev}'
    power_traces = qp.utils.powertrace_simple(trace=x[:,0,:], ioffset=ioffset, qetbias=qetbias, rload=rload, rsh=rsh)
    res = rp.downsample_truncate(traces=power_traces, trunc=power_traces.shape[-1], fs=fs, ds=ds_factor)
    power_ds = res['traces_ds']
    scale_factor = np.sum(np.sum(power_ds, axis = 1), axis=0)/power_ds.shape[-1]
    dump_number = path.split('/')[-1].split('.')[0]
    
    metadata = {'scale_factor' : scale_factor, 
                'n_traces' : x.shape[0],
                'dump' : dump_number}
    traces = {'traces' : power_ds[:,np.newaxis,:], 'eventnumber' : ser_ev}
    if lgcsave:
        vae.save_preprocessed(savepath, traces, metadata)
    return traces, metadata

