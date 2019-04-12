import pandas as pd
import numpy as np

def parse(fname,key,flatten=True, mode = '1d'):
    df = pd.read_hdf(fname,key=key,mode='r')
    
    traces= [t.reshape([-1]) if flatten else t for t in df['tm'].values]
    tms   = np.stack([t for t in traces]) 
    epos  = np.stack([[p[1], p[2], p[3], p[4], p[0]] for p in df['epos'].values])
    S     = np.stack([ ([p[-1]] if mode == '1d' else [p[-1]]*11) for p in df['epos'].values])
    return tms,epos,S
