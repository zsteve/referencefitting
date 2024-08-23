#####################################################
import autograd.numpy as np
# This file is created automatically
def Model(Y,t,pars):
    # Parameters
    a_Cebpa_Cebpa = pars[0]
    a_Cebpa_Cebpa_Fog1 = pars[1]
    a_Cebpa_Cebpa_Fog1_Gata1 = pars[2]
    a_Cebpa_Cebpa_Fog1_Gata1_Scl = pars[3]
    a_Cebpa_Cebpa_Fog1_Scl = pars[4]
    a_Cebpa_Cebpa_Gata1 = pars[5]
    a_Cebpa_Cebpa_Gata1_Scl = pars[6]
    a_Cebpa_Cebpa_Scl = pars[7]
    a_Cebpa_Fog1 = pars[8]
    a_Cebpa_Fog1_Gata1 = pars[9]
    a_Cebpa_Fog1_Gata1_Scl = pars[10]
    a_Cebpa_Fog1_Scl = pars[11]
    a_Cebpa_Gata1 = pars[12]
    a_Cebpa_Gata1_Scl = pars[13]
    a_Cebpa_Scl = pars[14]
    a_EgrNab_Gfi1 = pars[15]
    a_EgrNab_Pu1 = pars[16]
    a_EgrNab_Pu1_Gfi1 = pars[17]
    a_EgrNab_Pu1_cJun = pars[18]
    a_EgrNab_Pu1_cJun_Gfi1 = pars[19]
    a_EgrNab_cJun = pars[20]
    a_EgrNab_cJun_Gfi1 = pars[21]
    a_Eklf_Fli1 = pars[22]
    a_Eklf_Fli1_Gata1 = pars[23]
    a_Eklf_Gata1 = pars[24]
    a_Fli1_Eklf = pars[25]
    a_Fli1_Eklf_Gata1 = pars[26]
    a_Fli1_Gata1 = pars[27]
    a_Fog1_Gata1 = pars[28]
    a_Gata1_Fli1 = pars[29]
    a_Gata1_Fli1_Gata1 = pars[30]
    a_Gata1_Fli1_Gata2 = pars[31]
    a_Gata1_Fli1_Gata2_Gata1 = pars[32]
    a_Gata1_Fli1_Pu1 = pars[33]
    a_Gata1_Fli1_Pu1_Gata1 = pars[34]
    a_Gata1_Fli1_Pu1_Gata2 = pars[35]
    a_Gata1_Fli1_Pu1_Gata2_Gata1 = pars[36]
    a_Gata1_Gata1 = pars[37]
    a_Gata1_Gata2 = pars[38]
    a_Gata1_Gata2_Gata1 = pars[39]
    a_Gata1_Pu1 = pars[40]
    a_Gata1_Pu1_Gata1 = pars[41]
    a_Gata1_Pu1_Gata2 = pars[42]
    a_Gata1_Pu1_Gata2_Gata1 = pars[43]
    a_Gata2_Fog1 = pars[44]
    a_Gata2_Gata1 = pars[45]
    a_Gata2_Gata1_Fog1 = pars[46]
    a_Gata2_Gata2 = pars[47]
    a_Gata2_Gata2_Fog1 = pars[48]
    a_Gata2_Gata2_Gata1 = pars[49]
    a_Gata2_Gata2_Gata1_Fog1 = pars[50]
    a_Gata2_Pu1 = pars[51]
    a_Gata2_Pu1_Fog1 = pars[52]
    a_Gata2_Pu1_Gata1 = pars[53]
    a_Gata2_Pu1_Gata1_Fog1 = pars[54]
    a_Gata2_Pu1_Gata2 = pars[55]
    a_Gata2_Pu1_Gata2_Fog1 = pars[56]
    a_Gata2_Pu1_Gata2_Gata1 = pars[57]
    a_Gata2_Pu1_Gata2_Gata1_Fog1 = pars[58]
    a_Gfi1_Gfi1 = pars[59]
    a_Pu1_Cebpa = pars[60]
    a_Pu1_Cebpa_Gata1 = pars[61]
    a_Pu1_Cebpa_Gata2 = pars[62]
    a_Pu1_Cebpa_Gata2_Gata1 = pars[63]
    a_Pu1_Gata1 = pars[64]
    a_Pu1_Gata2 = pars[65]
    a_Pu1_Gata2_Gata1 = pars[66]
    a_Pu1_Pu1 = pars[67]
    a_Pu1_Pu1_Cebpa = pars[68]
    a_Pu1_Pu1_Cebpa_Gata1 = pars[69]
    a_Pu1_Pu1_Cebpa_Gata2 = pars[70]
    a_Pu1_Pu1_Cebpa_Gata2_Gata1 = pars[71]
    a_Pu1_Pu1_Gata1 = pars[72]
    a_Pu1_Pu1_Gata2 = pars[73]
    a_Pu1_Pu1_Gata2_Gata1 = pars[74]
    a_Scl_Gata1 = pars[75]
    a_Scl_Pu1 = pars[76]
    a_Scl_Pu1_Gata1 = pars[77]
    a_cJun_Gfi1 = pars[78]
    a_cJun_Pu1 = pars[79]
    a_cJun_Pu1_Gfi1 = pars[80]
    alpha_Cebpa = pars[81]
    alpha_EgrNab = pars[82]
    alpha_Eklf = pars[83]
    alpha_Fli1 = pars[84]
    alpha_Fog1 = pars[85]
    alpha_Gata1 = pars[86]
    alpha_Gata2 = pars[87]
    alpha_Gfi1 = pars[88]
    alpha_Pu1 = pars[89]
    alpha_Scl = pars[90]
    alpha_cJun = pars[91]
    k_Cebpa = pars[92]
    k_EgrNab = pars[93]
    k_Eklf = pars[94]
    k_Fli1 = pars[95]
    k_Fog1 = pars[96]
    k_Gata1 = pars[97]
    k_Gata2 = pars[98]
    k_Gfi1 = pars[99]
    k_Pu1 = pars[100]
    k_Scl = pars[101]
    k_cJun = pars[102]
    l_p_Cebpa = pars[103]
    l_p_EgrNab = pars[104]
    l_p_Eklf = pars[105]
    l_p_Fli1 = pars[106]
    l_p_Fog1 = pars[107]
    l_p_Gata1 = pars[108]
    l_p_Gata2 = pars[109]
    l_p_Gfi1 = pars[110]
    l_p_Pu1 = pars[111]
    l_p_Scl = pars[112]
    l_p_cJun = pars[113]
    l_x_Cebpa = pars[114]
    l_x_EgrNab = pars[115]
    l_x_Eklf = pars[116]
    l_x_Fli1 = pars[117]
    l_x_Fog1 = pars[118]
    l_x_Gata1 = pars[119]
    l_x_Gata2 = pars[120]
    l_x_Gfi1 = pars[121]
    l_x_Pu1 = pars[122]
    l_x_Scl = pars[123]
    l_x_cJun = pars[124]
    m_Cebpa = pars[125]
    m_EgrNab = pars[126]
    m_Eklf = pars[127]
    m_Fli1 = pars[128]
    m_Fog1 = pars[129]
    m_Gata1 = pars[130]
    m_Gata2 = pars[131]
    m_Gfi1 = pars[132]
    m_Pu1 = pars[133]
    m_Scl = pars[134]
    m_cJun = pars[135]
    n_Cebpa = pars[136]
    n_EgrNab = pars[137]
    n_Eklf = pars[138]
    n_Fli1 = pars[139]
    n_Fog1 = pars[140]
    n_Gata1 = pars[141]
    n_Gata2 = pars[142]
    n_Gfi1 = pars[143]
    n_Pu1 = pars[144]
    n_Scl = pars[145]
    n_cJun = pars[146]
    r_Cebpa = pars[147]
    r_EgrNab = pars[148]
    r_Eklf = pars[149]
    r_Fli1 = pars[150]
    r_Fog1 = pars[151]
    r_Gata1 = pars[152]
    r_Gata2 = pars[153]
    r_Gfi1 = pars[154]
    r_Pu1 = pars[155]
    r_Scl = pars[156]
    r_cJun = pars[157]
    sigmaH_Cebpa = pars[158]
    sigmaH_EgrNab = pars[159]
    sigmaH_Eklf = pars[160]
    sigmaH_Fli1 = pars[161]
    sigmaH_Fog1 = pars[162]
    sigmaH_Gata1 = pars[163]
    sigmaH_Gata2 = pars[164]
    sigmaH_Gfi1 = pars[165]
    sigmaH_Pu1 = pars[166]
    sigmaH_Scl = pars[167]
    sigmaH_cJun = pars[168]
    # Variables
    x_Gata2 = Y[0]
    p_Gata2 = Y[1]
    x_Gata1 = Y[2]
    p_Gata1 = Y[3]
    x_Fog1 = Y[4]
    p_Fog1 = Y[5]
    x_Eklf = Y[6]
    p_Eklf = Y[7]
    x_Fli1 = Y[8]
    p_Fli1 = Y[9]
    x_Scl = Y[10]
    p_Scl = Y[11]
    x_Cebpa = Y[12]
    p_Cebpa = Y[13]
    x_Pu1 = Y[14]
    p_Pu1 = Y[15]
    x_cJun = Y[16]
    p_cJun = Y[17]
    x_EgrNab = Y[18]
    p_EgrNab = Y[19]
    x_Gfi1 = Y[20]
    p_Gfi1 = Y[21]
    dx_Gata2 = m_Gata2*(( alpha_Gata2 + a_Gata2_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_Gata2_Gata2*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata2_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata2_Fog1*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Pu1_Gata2*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata2_Pu1_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata2_Pu1_Fog1*(p_Pu1/k_Pu1)**n_Pu1*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Gata2_Gata1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata2_Gata2_Fog1*(p_Gata2/k_Gata2)**n_Gata2*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Gata1_Fog1*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Pu1_Gata2_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata2_Pu1_Gata2_Fog1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Pu1_Gata1_Fog1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Gata2_Gata1_Fog1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 + a_Gata2_Pu1_Gata2_Gata1_Fog1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 )/( 1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_Gata2/k_Gata2)**n_Gata2 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Fog1/k_Fog1)**n_Fog1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Fog1/k_Fog1)**n_Fog1 +(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Gata2/k_Gata2)**n_Gata2*(p_Fog1/k_Fog1)**n_Fog1 +(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Fog1/k_Fog1)**n_Fog1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 +(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1*(p_Fog1/k_Fog1)**n_Fog1 ))-l_x_Gata2*x_Gata2
    dp_Gata2 = r_Gata2*x_Gata2- l_p_Gata2*p_Gata2
    dx_Gata1 = m_Gata1*(( alpha_Gata1 + a_Gata1_Fli1*(p_Fli1/k_Fli1)**n_Fli1 + a_Gata1_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_Gata1_Gata2*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata1_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Fli1_Pu1*(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1 + a_Gata1_Fli1_Gata2*(p_Fli1/k_Fli1)**n_Fli1*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata1_Fli1_Gata1*(p_Fli1/k_Fli1)**n_Fli1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Pu1_Gata2*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata1_Pu1_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Gata2_Gata1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Fli1_Pu1_Gata2*(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 + a_Gata1_Fli1_Pu1_Gata1*(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Fli1_Gata2_Gata1*(p_Fli1/k_Fli1)**n_Fli1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Pu1_Gata2_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Gata1_Fli1_Pu1_Gata2_Gata1*(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Fli1/k_Fli1)**n_Fli1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_Gata2/k_Gata2)**n_Gata2 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Gata2/k_Gata2)**n_Gata2 +(p_Fli1/k_Fli1)**n_Fli1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 +(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Gata1*x_Gata1
    dp_Gata1 = r_Gata1*x_Gata1- l_p_Gata1*p_Gata1
    dx_Fog1 = m_Fog1*(( alpha_Fog1 + a_Fog1_Gata1*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Fog1*x_Fog1
    dp_Fog1 = r_Fog1*x_Fog1- l_p_Fog1*p_Fog1
    dx_Eklf = m_Eklf*(( alpha_Eklf + a_Eklf_Fli1*(p_Fli1/k_Fli1)**n_Fli1 + a_Eklf_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Eklf_Fli1_Gata1*(p_Fli1/k_Fli1)**n_Fli1*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Fli1/k_Fli1)**n_Fli1 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Fli1/k_Fli1)**n_Fli1*(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Eklf*x_Eklf
    dp_Eklf = r_Eklf*x_Eklf- l_p_Eklf*p_Eklf
    dx_Fli1 = m_Fli1*(( alpha_Fli1 + a_Fli1_Eklf*(p_Eklf/k_Eklf)**n_Eklf + a_Fli1_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Fli1_Eklf_Gata1*(p_Eklf/k_Eklf)**n_Eklf*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Eklf/k_Eklf)**n_Eklf +(p_Gata1/k_Gata1)**n_Gata1 +(p_Eklf/k_Eklf)**n_Eklf*(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Fli1*x_Fli1
    dp_Fli1 = r_Fli1*x_Fli1- l_p_Fli1*p_Fli1
    dx_Scl = m_Scl*(( alpha_Scl + a_Scl_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_Scl_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Scl_Pu1_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Scl*x_Scl
    dp_Scl = r_Scl*x_Scl- l_p_Scl*p_Scl
    dx_Cebpa = m_Cebpa*(( alpha_Cebpa + a_Cebpa_Cebpa*(p_Cebpa/k_Cebpa)**n_Cebpa + a_Cebpa_Fog1*(p_Fog1/k_Fog1)**n_Fog1 + a_Cebpa_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Cebpa_Scl*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Cebpa_Fog1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1 + a_Cebpa_Cebpa_Gata1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 + a_Cebpa_Cebpa_Scl*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Fog1_Gata1*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1 + a_Cebpa_Fog1_Scl*(p_Fog1/k_Fog1)**n_Fog1*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Gata1_Scl*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Cebpa_Fog1_Gata1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1 + a_Cebpa_Cebpa_Fog1_Scl*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Cebpa_Gata1_Scl*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Fog1_Gata1_Scl*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl + a_Cebpa_Cebpa_Fog1_Gata1_Scl*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl )/( 1 +(p_Cebpa/k_Cebpa)**n_Cebpa +(p_Fog1/k_Fog1)**n_Fog1 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Scl/k_Scl)**n_Scl +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Scl/k_Scl)**n_Scl +(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Fog1/k_Fog1)**n_Fog1*(p_Scl/k_Scl)**n_Scl +(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Scl/k_Scl)**n_Scl +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl +(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Fog1/k_Fog1)**n_Fog1*(p_Gata1/k_Gata1)**n_Gata1*(p_Scl/k_Scl)**n_Scl ))-l_x_Cebpa*x_Cebpa
    dp_Cebpa = r_Cebpa*x_Cebpa- l_p_Cebpa*p_Cebpa
    dx_Pu1 = m_Pu1*(( alpha_Pu1 + a_Pu1_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_Pu1_Cebpa*(p_Cebpa/k_Cebpa)**n_Cebpa + a_Pu1_Gata2*(p_Gata2/k_Gata2)**n_Gata2 + a_Pu1_Gata1*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Pu1_Cebpa*(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa + a_Pu1_Pu1_Gata2*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 + a_Pu1_Pu1_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Cebpa_Gata2*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2 + a_Pu1_Cebpa_Gata1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Gata2_Gata1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Pu1_Cebpa_Gata2*(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2 + a_Pu1_Pu1_Cebpa_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Pu1_Gata2_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Cebpa_Gata2_Gata1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 + a_Pu1_Pu1_Cebpa_Gata2_Gata1*(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 )/( 1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_Cebpa/k_Cebpa)**n_Cebpa +(p_Gata2/k_Gata2)**n_Gata2 +(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata1/k_Gata1)**n_Gata1 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 +(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2 +(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Cebpa/k_Cebpa)**n_Cebpa*(p_Gata2/k_Gata2)**n_Gata2*(p_Gata1/k_Gata1)**n_Gata1 ))-l_x_Pu1*x_Pu1
    dp_Pu1 = r_Pu1*x_Pu1- l_p_Pu1*p_Pu1
    dx_cJun = m_cJun*(( alpha_cJun + a_cJun_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_cJun_Gfi1*(p_Gfi1/k_Gfi1)**n_Gfi1 + a_cJun_Pu1_Gfi1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gfi1/k_Gfi1)**n_Gfi1 )/( 1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_Gfi1/k_Gfi1)**n_Gfi1 +(p_Pu1/k_Pu1)**n_Pu1*(p_Gfi1/k_Gfi1)**n_Gfi1 ))-l_x_cJun*x_cJun
    dp_cJun = r_cJun*x_cJun- l_p_cJun*p_cJun
    dx_EgrNab = m_EgrNab*(( alpha_EgrNab + a_EgrNab_Pu1*(p_Pu1/k_Pu1)**n_Pu1 + a_EgrNab_cJun*(p_cJun/k_cJun)**n_cJun + a_EgrNab_Gfi1*(p_Gfi1/k_Gfi1)**n_Gfi1 + a_EgrNab_Pu1_cJun*(p_Pu1/k_Pu1)**n_Pu1*(p_cJun/k_cJun)**n_cJun + a_EgrNab_Pu1_Gfi1*(p_Pu1/k_Pu1)**n_Pu1*(p_Gfi1/k_Gfi1)**n_Gfi1 + a_EgrNab_cJun_Gfi1*(p_cJun/k_cJun)**n_cJun*(p_Gfi1/k_Gfi1)**n_Gfi1 + a_EgrNab_Pu1_cJun_Gfi1*(p_Pu1/k_Pu1)**n_Pu1*(p_cJun/k_cJun)**n_cJun*(p_Gfi1/k_Gfi1)**n_Gfi1 )/( 1 +(p_Pu1/k_Pu1)**n_Pu1 +(p_cJun/k_cJun)**n_cJun +(p_Gfi1/k_Gfi1)**n_Gfi1 +(p_Pu1/k_Pu1)**n_Pu1*(p_cJun/k_cJun)**n_cJun +(p_Pu1/k_Pu1)**n_Pu1*(p_Gfi1/k_Gfi1)**n_Gfi1 +(p_cJun/k_cJun)**n_cJun*(p_Gfi1/k_Gfi1)**n_Gfi1 +(p_Pu1/k_Pu1)**n_Pu1*(p_cJun/k_cJun)**n_cJun*(p_Gfi1/k_Gfi1)**n_Gfi1 ))-l_x_EgrNab*x_EgrNab
    dp_EgrNab = r_EgrNab*x_EgrNab- l_p_EgrNab*p_EgrNab
    dx_Gfi1 = m_Gfi1*(( alpha_Gfi1 + a_Gfi1_Gfi1*(p_Gfi1/k_Gfi1)**n_Gfi1 )/( 1 +(p_Gfi1/k_Gfi1)**n_Gfi1 ))-l_x_Gfi1*x_Gfi1
    dp_Gfi1 = r_Gfi1*x_Gfi1- l_p_Gfi1*p_Gfi1
    dY = np.array([dx_Gata2,dp_Gata2,dx_Gata1,dp_Gata1,dx_Fog1,dp_Fog1,dx_Eklf,dp_Eklf,dx_Fli1,dp_Fli1,dx_Scl,dp_Scl,dx_Cebpa,dp_Cebpa,dx_Pu1,dp_Pu1,dx_cJun,dp_cJun,dx_EgrNab,dp_EgrNab,dx_Gfi1,dp_Gfi1,])
    return(dY)
#####################################################