###############################################################
# Author : Caleb Fink
# 5/9/19
#
# This file contains usefull globa paths to processed and raw 
# data, as well as all the names of the labels for the calibrated
# PD2 dataset. All are available from base import
###############################################################

TRACE_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v2/traces/'
META_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v2/metadata/'
LABEL_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v2/labels/'

TRACE_PATH_DS = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v3/traces/'
META_PATH_DS = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v3/metadata/'
LABEL_PATH_DS = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_v3/labels/'

RQ_DF_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/Run44_processed_DF.pkl'
EVENT_FILE_MAP_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/file_mapping.h5'
PARTITION_PATH = '/gpfs/slac/staas/fs1/supercdms/tf/slac/Run44/partitions/'

PD2_LABEL_COLUMNS = ['eventnumber', 'seriesnumber', 'eventtime', 'triggertype',
       'pollingendtime', 'triggertime', 'triggeramp', 'readoutstatusZ1',
       'deadtimeZ1', 'livetimeZ1', 'triggervetoreadouttimeZ1',
       'seriestimeZ1', 'waveformreadendtimeZ1', 'waveformreadstarttimeZ1',
       'baseline_PDS1Z1', 'integral_PDS1Z1', 'energy_absorbed_PDS1Z1',
       'maxmin_PDS1Z1', 'chi2_nopulse_PDS1Z1', 'ofamp_nodelay_PDS1Z1',
       'chi2_nodelay_PDS1Z1', 'ofamp_unconstrain_PDS1Z1',
       't0_unconstrain_PDS1Z1', 'chi2_unconstrain_PDS1Z1',
       'ofamp_unconstrain_pcon_PDS1Z1', 't0_unconstrain_pcon_PDS1Z1',
       'chi2_unconstrain_pcon_PDS1Z1', 'ofamp_constrain_PDS1Z1',
       't0_constrain_PDS1Z1', 'chi2_constrain_PDS1Z1',
       'ofamp_constrain_pcon_PDS1Z1', 't0_constrain_pcon_PDS1Z1',
       'chi2_constrain_pcon_PDS1Z1', 'chi2lowfreq_nodelay_PDS1Z1',
       'chi2lowfreq_unconstrain_PDS1Z1',
       'chi2lowfreq_unconstrain_pcon_PDS1Z1',
       'chi2lowfreq_constrain_PDS1Z1',
       'chi2lowfreq_constrain_pcon_PDS1Z1', 'ofamp_pileup_PDS1Z1',
       't0_pileup_PDS1Z1', 'chi2_pileup_PDS1Z1',
       'ofamp_pileup_pcon_PDS1Z1', 't0_pileup_pcon_PDS1Z1',
       'chi2_pileup_pcon_PDS1Z1', 'ofamp_nlin_PDS1Z1',
       'ofamp_nlin_err_PDS1Z1', 'oftaurise_nlin_PDS1Z1',
       'oftaurise_nlin_err_PDS1Z1', 'oftaufall_nlin_PDS1Z1',
       'oftaufall_nlin_err_PDS1Z1', 't0_nlin_PDS1Z1',
       't0_nlin_err_PDS1Z1', 'chi2_nlin_PDS1Z1',
       'ofenergy_nodelay_full_PDS1Z1', 'ofenergy_nodelay_err_full_PDS1Z1',
       'ofenergy_constrain_full_PDS1Z1',
       'ofenergy_constrain_err_full_PDS1Z1',
       'ofenergy_unconstrain_full_PDS1Z1',
       'ofenergy_unconstrain_err_full_PDS1Z1',
       'ofenergy_constrain_pcon_full_PDS1Z1',
       'ofenergy_constrain_pcon_err_full_PDS1Z1',
       'ofenergy_unconstrain_pcon_full_PDS1Z1',
       'ofenergy_unconstrain_pcon_err_full_PDS1Z1',
       'integral_energy_full_PDS1Z1', 'integral_energy_err_full_PDS1Z1',
       'ofenergy_nodelay_al_PDS1Z1', 'ofenergy_nodelay_err_al_PDS1Z1',
       'ofenergy_constrain_al_PDS1Z1', 'ofenergy_constrain_err_al_PDS1Z1',
       'ofenergy_unconstrain_al_PDS1Z1',
       'ofenergy_unconstrain_err_al_PDS1Z1',
       'ofenergy_constrain_pcon_al_PDS1Z1',
       'ofenergy_constrain_pcon_err_al_PDS1Z1',
       'ofenergy_unconstrain_pcon_al_PDS1Z1',
       'ofenergy_unconstrain_pcon_err_al_PDS1Z1',
       'integral_energy_al_PDS1Z1', 'integral_energy_err_al_PDS1Z1',
       'ofenergy_nodelay_int_full_PDS1Z1',
       'ofenergy_nodelay_err_int_full_PDS1Z1',
       'ofenergy_constrain_int_full_PDS1Z1',
       'ofenergy_constrain_err_int_full_PDS1Z1',
       'ofenergy_unconstrain_int_full_PDS1Z1',
       'ofenergy_unconstrain_err_int_full_PDS1Z1',
       'ofenergy_constrain_pcon_int_full_PDS1Z1',
       'ofenergy_constrain_pcon_err_int_full_PDS1Z1',
       'ofenergy_unconstrain_pcon_int_full_PDS1Z1',
       'ofenergy_unconstrain_pcon_err_int_full_PDS1Z1',
       'integral_energy_int_full_PDS1Z1',
       'integral_energy_err_int_full_PDS1Z1',
       'ofenergy_nodelay_al_int_PDS1Z1',
       'ofenergy_nodelay_err_al_int_PDS1Z1',
       'ofenergy_constrain_al_int_PDS1Z1',
       'ofenergy_constrain_err_al_int_PDS1Z1',
       'ofenergy_unconstrain_al_int_PDS1Z1',
       'ofenergy_unconstrain_err_al_int_PDS1Z1',
       'ofenergy_constrain_pcon_al_int_PDS1Z1',
       'ofenergy_constrain_pcon_err_al_int_PDS1Z1',
       'ofenergy_unconstrain_pcon_al_int_PDS1Z1',
       'ofenergy_unconstrain_pcon_err_al_int_PDS1Z1',
       'integral_energy_al_int_PDS1Z1',
       'integral_energy_err_al_int_PDS1Z1', 'ser_ev', 'eventseriesnumber',
       'energy']