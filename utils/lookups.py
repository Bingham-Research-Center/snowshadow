"""Dictionary of kernel sizes for each site to smooth snow data.
"""
import os

kernel_lookup = {
    ##### CSLV #####
    "HERUT": 301,  # Good time series; high frequency
    # "NHMU": 301, # REMOVE
    # "SUNUT": 301, # REMOVE
    "SNV": 301,  # Good time series; high frequency
    "DVO": 501,  # TODO manually remove end of 2023 with big dropout

    "LPTUT": 301,  # High res, good ts,
    "ELBUT": 301,  # TODO High res, good ts, a square wave at start of 2024
    "SPC": 501,  # TODO High res, good ts, end of 2023 has dropout
    "HOL": 101,  # Low values but long ts, high freq, low noise
    # "PNCUT": 1, # REMOVE

    "CLN": 201,  # TODO High freq with min value noise - remove end of 2023 dropout
    "FMNU1": 201,  # Good ts, high freq
    "THCU1": 101,  # Good ts, high freq
    "MLDU1": 301,  # Good ts, high freq
    "CDYBK": 101,  # Good ts, high freq

    ##### NSLV #####
    "UTPW2": 301,  # very nois
    # "FMNU1": 1, # DUPLICATE
    "COOPOGNU1": 1,  # not noisy
    "PWDU1": 201,  # rather nosiy
    "SNFEC": 401,  # very noisy

    # "BLPU1": 301, # lots of weird bits - REMOVE
    # "FARU1": 301, # lots of weird bits - REMOVE
    "COOPBRGU1": 1,  # smooth, lower freq
    "SNI": 101,  # TODO high freq, some noise - remove dropout at end of 2023
    "PCRU1": 101,  # high freq, some noise

    # "BLTU1": 301, # too much erroneous data - REMOVE
    "LTBU1": 101,  # high freq, some noise
    "COOPWEBU1": 1,  # smooth, lower freq - potentially a gap in early 2024 season but low freq anyway
    "SBBWK": 301,  # TODO very noisy - chop beginning and end of both years that are missing/erroneous
    "TPR": 101,  # high freq, some noise

    ##### SSLV #####
    # "SUNUT": 1001, # DUPLICATE
    # "LPTUT": 1, # DUPLICATE
    # "UTASG": 1001, # very noisy - REMOVE
    "BUNUT": 501,  # very noisy TODO - remove end of both years dropout
    # "PNCUT": 1, # Missing 2023?

    "TIMU1": 501,  # very noisy, high freq
    "COOPPLTU1": 1,
    "COOPAMFU1": 1,
    "COOPPROU1": 1,
    "COOPLEHU1": 1,

    "COOPPLGU1": 1,
    "COOPFAFU1": 1,
    "SNM": 201,  # noisy
    "DCC": 201,  # noisy
    "PYSU1": 201,  # noisy

    ##### UINTA BASIN #####
    "COOPNELU1": 1,
    "COOPALMU1": 1,
    "COOPJENU1": 1,
    "COOPFTDU1": 3,  # A little noise
    "MMTU1": 151,  # noisy

    "COOPROSU1": 1,
    "COOPDSNU1": 1,
    "COOPMYTU1": 1,
    "KGCU1": 301,  # very noisy
    # "SPKU1": 301 # very noisy - REMOVE too much missing/erroneous

    "TCKU1": 301,  # noisy
    "LKFU1": 1101,
    "CWHU1": 601,
    "COOPOURU1": 1,
    "UTWIS": 501,

    "TPKUT": 101,
    "CSTUT": 101,
    # "GSTPS": 101, # weird data - removed
    "CCSUT": 101,
    # "TPRUT": 1, # weird data - removed

    # "COOPNELU1": 1, # duplicate
    # "COOPALMU1": 1, # duplicate
    "CCKU1": 301,
    "HPSU1": 51,
    "HIRU1": 51,

    "LLKU1": 101,
    "BCZU1": 101,
    "HEWU1": 101,
    # "CHCU1": 101, # REMOVE too much weird data
    "RMLU1": 101,

}