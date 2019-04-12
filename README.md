# VAE

Collection of tools for testing the idea of using a Variational autoencoder for event reconstruction. The directory `working_dir` holds various working files and notebooks, and the directory `vae` holds the VAE module tools. 

### installation

To install and use these tools, from the top level directory do the following in the command line:

```
>>> python setup.py clean
>>> python setup.py install --user
```

Among the standard packages included with Anaconda, the following dependencies are needed: `deepdish`, `QETpy`, and `RQpy`. Most of these can be installed via

```
>>> pip install -r requirements.txt
```
Except `RQpy` which is not yet avaliable on PyPi. To install `RQpy` clone the repository https://github.com/ucbpylegroup/RQpy and follow the install instructions in the README.md. It is also recommended that the most current development version of `QETpy` is used, which can be cloned and installed from https://github.com/ucbpylegroup/QETpy

### Usage 

At the moment, all of the functions (only a few at the moment) are available from the top level directory. For example,
```python
import vae

traces, metadata = vae.pre_process_PD2(path=files[ii], fs=fs, ioffset=ioffset, rload=rload, 
                                       rsh=rsh, qetbias=qetbias, chan=chan, det=det, trunc=14000)
```

                                       
                                

