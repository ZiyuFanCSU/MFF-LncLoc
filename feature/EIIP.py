import codecs
import numpy as np
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
def EIIP_pos(filename):  # ROOT_DIR+'/case_feature.txt'
    f = codecs.open((filename), mode='r', encoding='utf-8')
    line = f.readline()
    lnc_dist, pct_dist, dist_ratio, peak, SNR, min, Q1, Q2, max= [], [], [], [], [], [], [], [], []
    dot_lnc, dot_pct, dot_ratio, ss_lnc, ss_pct, ss_ratio, MFE, up_pct = [], [], [], [], [], [], [], []
    flag = 0
    while line:
        if flag == 0:
            line = f.readline()
            flag += 1
        else:
            a = line.split()
            tmp = []
            for i,j in enumerate(range(5, 22)):
                tmp.append(float(a[j: j+1][0]))
            lnc_dist.append(tmp[0])
            pct_dist.append(tmp[1])
            dist_ratio.append(tmp[2])
            peak.append(tmp[3])
            SNR.append(tmp[4])
            min.append(tmp[5])
            Q1.append(tmp[6])
            Q2.append(tmp[7])
            max.append(tmp[8])
            dot_lnc.append(tmp[9])
            dot_pct.append(tmp[10])
            dot_ratio.append(tmp[11])
            ss_lnc.append(tmp[12])
            ss_pct.append(tmp[13])
            ss_ratio.append(tmp[14])
            MFE.append(tmp[15])
            up_pct.append(tmp[16])
            line = f.readline()
    f.close()
 
    eiip_pos = [lnc_dist, pct_dist, dist_ratio, peak, SNR, dot_lnc, dot_pct, dot_ratio, ss_lnc, ss_pct, ss_ratio, MFE, up_pct]
    # eiip_pos = [lnc_dist, pct_dist, dist_ratio, peak, SNR, MFE]
    eiip_pos = np.transpose(eiip_pos).tolist()
    return eiip_pos


def EIIP_neg(filename):   # ROOT_DIR+'/subloc_feature.txt'
    f = codecs.open((filename), mode='r', encoding='utf-8')
    line = f.readline()
    lnc_dist, pct_dist, dist_ratio, peak, SNR, min, Q1, Q2, max= [], [], [], [], [], [], [], [], []
    flag = 0
    while line:
        if flag == 0:
            line = f.readline()
            flag += 1
        else:
            a = line.split()
            tmp = []
            for i,j in enumerate(range(5, 14)):
                tmp.append(float(a[j: j+1][0]))
            lnc_dist.append(tmp[0])
            pct_dist.append(tmp[1])
            dist_ratio.append(tmp[2])
            peak.append(tmp[3])
            SNR.append(tmp[4])
            min.append(tmp[5])
            Q1.append(tmp[6])
            Q2.append(tmp[7])
            max.append(tmp[8])
            line = f.readline()
    f.close()
    # eiip_neg = [lnc_dist, pct_dist, dist_ratio]
    # eiip_neg = [peak, SNR]
    eiip_neg = [lnc_dist, pct_dist,dist_ratio, peak, SNR, min, Q1, Q2, max]
    eiip_neg = np.transpose(eiip_neg).tolist()
    return eiip_neg
