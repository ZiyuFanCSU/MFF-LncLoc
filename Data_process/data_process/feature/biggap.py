import numpy as np


def big_gap_single(seq, ggaparray, g):
    # seq length is fix =23

    rst = np.zeros((256))
    for i in range(len(seq) - 1 - g):
        str1 = seq[i] + seq[i + 1]
        str2 = seq[i + g] + seq[i + 1 + g]
        idx = ggaparray.index(str1 + str2)
        rst[idx] += 1

    for j in range(len(ggaparray)):
        rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

    return rst


def construct_kmer():
    ntarr = ("A", "C", "G", "T")

    kmerArray = []

    for n in range(4):
        kmerArray.append(ntarr[n])

    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            kmerArray.append(str2)
    #############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                kmerArray.append(str3)
    #############################################
    # change this part for 3mer or 4mer
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    kmerArray.append(str4)
    ############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    for z in range(4):
                        str5 = str4 + ntarr[z]
                        kmerArray.append(str5)
    ####################### 6-mer ##############
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    for z in range(4):
                        str5 = str4 + ntarr[z]
                        for t in range(4):
                            str6 = str5 + ntarr[t]
                            kmerArray.append(str6)
    ####################### 7-mer ##############
    kmer7 = []
    for m in kmerArray[1364:5460]:
        for i in ntarr:
            st7 = m + i
            kmer7.append(st7)
    kmerArray = kmerArray + kmer7

    return kmerArray

def biggap_encode(seq,ggaparray,g):
    result = []
    for x in seq:
        temp = big_gap_single(x,ggaparray,g)
        result.append(temp)
    result = np.array(result)
    return result

def encode(RNA_seq):
    kmerArray = construct_kmer()
    # print(kmerArray)

    RNA_biggap2 = biggap_encode(RNA_seq,kmerArray[84:340],2)
    RNA_biggap3 = biggap_encode(RNA_seq,kmerArray[84:340],3)

    return RNA_biggap2

seq = 'AAGGTTTGTGTTTCTTTGGGCAGGAGCCTACTGTGATCAGAGCTGGGGGAGCCAGTGTGAGCAGCAGGTGCAATGAAAGGATGTGGGAGTTTGAGACCCATCTCTGCTGTGTGGCCCTGGCCATGTGCAGTGGGTAAGTACATCAGCAAGCAAACTGGCCTCAAGAATCCCGCCCTATCCTGGGCATCTGGGGTTGAGGCTGGCTGTGCCAACCTTTGCACAGGGTTTCAATCTGACCTCACTCAGAAAACGGAGCCACTTGCTTAGAGTCCCTCTCTCTGAACACGCTCTCCCCTCTGGAGGAAAAGCATTGCAATCCCAAGCTCCTGCTGCTGCCGTACATCTGCATCTTATTCATCTTTGAAGCTCCTGCAATGCCTCCAACGGGGACACAACTAAAGGAAGCTCTCAGAACGCCCAGCTATCCCCACTGCAGAAGGCTCTCCCTGAAAGGGCACAGGTGGAGGACGTTTGCATTTATCTGGAGACAGTTCATGGAGAAACACAAAGAGGTGTCCTGACAAGACAACAGGTCCACAGCAGTCGTCATCATGAGTGCTCACCATATGCCAGGGTTCATCTTTTAAAATCCTTAACCAAACAACCAAACTGTGGTTGGGATTATTCTCTTTTTATAGGTGAGAAAACTGAGGCTGAAAGCCATTGGGTGGTTTGCCAAAGCTGAGAAGTAGCTGAGCTGGGGTTTGTACTCACACCACGGTGACTCGGAGGCCCAAGCCAGTTCTGACACACTGCCTTCCCATGACATTCCCACGACAGGTTCCCGAGGTCGTTGGAATGCTTACACATTTCAGCAACTTCATGGCAACTGCCCAGCTCGACCCAGAAGATAATACTACCTTGTACTCCCCCTGCCGCTTCCTTCCTCCAGGCAGCTCACAGACATGACCTCATTCCCTCTCCCAGAGTCCCATAGGGGTGAAGGACAGGGTGGGGATCGTTAACTCTGCTTCCTAGGGAGGAAAGGGAGGCCCAGAGAGGGATGCTGGCCCACCCTGGGTCATCCAGCCAGCTAGCGGGAGAGCTGCAACTGGAAATCAGGCTTCCCTGCCCTCGGGGACAGTTATTTAGCCTCCTGGATGTGAGCTAGAATCACCCCTAAGATGCAGCAATTGGGGTACTCCTCTTGGGGTACTGGGACTCAGTGGCTGAAACAGCTCCAAGAGAGTTGGGCCTGTGCTGAAAACAGAAATCCGAGATGCATATAAATTCAATCCGTATGAGAAGAGCAGGGTGGCCTGGGAGGAGTGGTCACCGCAGTGGCTCCACCTCCTACCCGGCCACCTGCAGCCCTGCCTGGGCTGCTCTTGGTCTCTCTGTGCCCAGCTTCCCTGGCCGGATGCCCTGGGGGCCAGAGCAGCCGCAGGTGCGATGAAAAGAACACTGGCCTTGAGGTCATGGGACCTGGGGAGGATATTCACCAGGGGCCCCTAAGTTAGTCATTTTAGCTCTCAGCTTTTCCATTTTTCATTCAAAAAATGGGGACTCTAACTCCATGAAAGAGACATTGTGTGTGAAATATTTCATCATAAGCTGTGAAGCCTGCAAAGAAACTCCTGGTGTTCTCTGTAGCCTGATTTCATCTCTACTAAACGCTATTTTGAAAATG'
seq_c = 'ATGGCTGGGATCACCACCATCGAGGCAGTGAAGCGCAAGATCCAGGTTCTGCAGCTGCAGGCAGATGATGAGGAGTGAGCTGAGCACCTCCAGTGAGAAGCTGAGAGAGAAAGGTGGGCCCGGGAACAGGCTGAGGCTGAAGTGGCCTCCGTGAACGGTAGGATCCAGCTGGTTGAAGAGGAGCTGGACTGTGCTCAGGAGCGCCTGGCCACTGCCCTGCAAAAGCTGGAAGAAGCGGGAAAAGCTGCTGATGAGAGTGAGAGAGATACAAAGGTTATTGAAATCTGGGCCTTAAAAGATGAAGAAGATGGAACTCCAGGAAATCCAACTCAAAGAAGCTAAGCACATTGCAGATGAGGCAGATGGGAAGTATGAAGAGGTGGCTCGTAAGTTGGTGATCATTGAAAGAGACATGGAATGCACAGAAGAATGAGCTGAGCTGGCAGAGTCCCGTTGCTGAGAGATGGATGAGCAGATCAGACTGATGGACCAGAACCTGAAGTGTCTGAGTGCAGCTGAAGAAAAGTACCCTCAAAAAGAAGACAAATGTGAGGAAGAGATGAAGATTCTTACTGATAATCTCGAGGAGGCAGAGACCCATGCTGAGTTGGCTGAGAGATCAGTAGCCAAGCTGGAAAAGACAATTGATGACTTGGAAGATAAACTGAAATGCACCAAAGAGGAACACCTCTGTACACAAAGGATGCTGGACCAGACTCTGCTTGACCTGAATGAGATGTAG'
print(encode(seq_c))