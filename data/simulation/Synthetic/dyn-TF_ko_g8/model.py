#####################################################
import autograd.numpy as np
# This file is created automatically
def Model(Y,t,pars):
    # Parameters
    a_g1_g4 = pars[0]
    a_g1_g4_g5 = pars[1]
    a_g1_g4_g5_g6 = pars[2]
    a_g1_g4_g6 = pars[3]
    a_g1_g5 = pars[4]
    a_g1_g5_g6 = pars[5]
    a_g1_g6 = pars[6]
    a_g2_g1 = pars[7]
    a_g3_g2 = pars[8]
    a_g4_g3 = pars[9]
    a_g4_g4 = pars[10]
    a_g4_g4_g3 = pars[11]
    a_g4_g4_g5 = pars[12]
    a_g4_g4_g5_g3 = pars[13]
    a_g4_g4_g6 = pars[14]
    a_g4_g4_g6_g3 = pars[15]
    a_g4_g4_g6_g5 = pars[16]
    a_g4_g4_g6_g5_g3 = pars[17]
    a_g4_g5 = pars[18]
    a_g4_g5_g3 = pars[19]
    a_g4_g6 = pars[20]
    a_g4_g6_g3 = pars[21]
    a_g4_g6_g5 = pars[22]
    a_g4_g6_g5_g3 = pars[23]
    a_g5_g3 = pars[24]
    a_g5_g4 = pars[25]
    a_g5_g4_g3 = pars[26]
    a_g5_g4_g5 = pars[27]
    a_g5_g4_g5_g3 = pars[28]
    a_g5_g4_g6 = pars[29]
    a_g5_g4_g6_g3 = pars[30]
    a_g5_g4_g6_g5 = pars[31]
    a_g5_g4_g6_g5_g3 = pars[32]
    a_g5_g5 = pars[33]
    a_g5_g5_g3 = pars[34]
    a_g5_g6 = pars[35]
    a_g5_g6_g3 = pars[36]
    a_g5_g6_g5 = pars[37]
    a_g5_g6_g5_g3 = pars[38]
    a_g6_g3 = pars[39]
    a_g6_g3_g7 = pars[40]
    a_g6_g4 = pars[41]
    a_g6_g4_g3 = pars[42]
    a_g6_g4_g3_g7 = pars[43]
    a_g6_g4_g5 = pars[44]
    a_g6_g4_g5_g3 = pars[45]
    a_g6_g4_g5_g3_g7 = pars[46]
    a_g6_g4_g5_g6 = pars[47]
    a_g6_g4_g5_g6_g3 = pars[48]
    a_g6_g4_g5_g6_g3_g7 = pars[49]
    a_g6_g4_g5_g6_g7 = pars[50]
    a_g6_g4_g5_g7 = pars[51]
    a_g6_g4_g6 = pars[52]
    a_g6_g4_g6_g3 = pars[53]
    a_g6_g4_g6_g3_g7 = pars[54]
    a_g6_g4_g6_g7 = pars[55]
    a_g6_g4_g7 = pars[56]
    a_g6_g5 = pars[57]
    a_g6_g5_g3 = pars[58]
    a_g6_g5_g3_g7 = pars[59]
    a_g6_g5_g6 = pars[60]
    a_g6_g5_g6_g3 = pars[61]
    a_g6_g5_g6_g3_g7 = pars[62]
    a_g6_g5_g6_g7 = pars[63]
    a_g6_g5_g7 = pars[64]
    a_g6_g6 = pars[65]
    a_g6_g6_g3 = pars[66]
    a_g6_g6_g3_g7 = pars[67]
    a_g6_g6_g7 = pars[68]
    a_g6_g7 = pars[69]
    a_g7_g6 = pars[70]
    a_g8_g8 = pars[71]
    alpha_g1 = pars[72]
    alpha_g2 = pars[73]
    alpha_g3 = pars[74]
    alpha_g4 = pars[75]
    alpha_g5 = pars[76]
    alpha_g6 = pars[77]
    alpha_g7 = pars[78]
    alpha_g8 = pars[79]
    k_g1 = pars[80]
    k_g2 = pars[81]
    k_g3 = pars[82]
    k_g4 = pars[83]
    k_g5 = pars[84]
    k_g6 = pars[85]
    k_g7 = pars[86]
    k_g8 = pars[87]
    l_p_g1 = pars[88]
    l_p_g2 = pars[89]
    l_p_g3 = pars[90]
    l_p_g4 = pars[91]
    l_p_g5 = pars[92]
    l_p_g6 = pars[93]
    l_p_g7 = pars[94]
    l_p_g8 = pars[95]
    l_x_g1 = pars[96]
    l_x_g2 = pars[97]
    l_x_g3 = pars[98]
    l_x_g4 = pars[99]
    l_x_g5 = pars[100]
    l_x_g6 = pars[101]
    l_x_g7 = pars[102]
    l_x_g8 = pars[103]
    m_g1 = pars[104]
    m_g2 = pars[105]
    m_g3 = pars[106]
    m_g4 = pars[107]
    m_g5 = pars[108]
    m_g6 = pars[109]
    m_g7 = pars[110]
    m_g8 = pars[111]
    n_g1 = pars[112]
    n_g2 = pars[113]
    n_g3 = pars[114]
    n_g4 = pars[115]
    n_g5 = pars[116]
    n_g6 = pars[117]
    n_g7 = pars[118]
    n_g8 = pars[119]
    r_g1 = pars[120]
    r_g2 = pars[121]
    r_g3 = pars[122]
    r_g4 = pars[123]
    r_g5 = pars[124]
    r_g6 = pars[125]
    r_g7 = pars[126]
    r_g8 = pars[127]
    sigmaH_g1 = pars[128]
    sigmaH_g2 = pars[129]
    sigmaH_g3 = pars[130]
    sigmaH_g4 = pars[131]
    sigmaH_g5 = pars[132]
    sigmaH_g6 = pars[133]
    sigmaH_g7 = pars[134]
    sigmaH_g8 = pars[135]
    # Variables
    x_g2 = Y[0]
    p_g2 = Y[1]
    x_g3 = Y[2]
    p_g3 = Y[3]
    x_g4 = Y[4]
    p_g4 = Y[5]
    x_g1 = Y[6]
    p_g1 = Y[7]
    x_g5 = Y[8]
    p_g5 = Y[9]
    x_g6 = Y[10]
    p_g6 = Y[11]
    x_g7 = Y[12]
    p_g7 = Y[13]
    x_g8 = Y[14]
    p_g8 = Y[15]
    dx_g2 = m_g2*(( alpha_g2 + a_g2_g1*(p_g1/k_g1)**n_g1 )/( 1 +(p_g1/k_g1)**n_g1 ))-l_x_g2*x_g2
    dp_g2 = r_g2*x_g2- l_p_g2*p_g2
    dx_g3 = m_g3*(( alpha_g3 + a_g3_g2*(p_g2/k_g2)**n_g2 )/( 1 +(p_g2/k_g2)**n_g2 ))-l_x_g3*x_g3
    dp_g3 = r_g3*x_g3- l_p_g3*p_g3
    dx_g4 = m_g4*(( alpha_g4 + a_g4_g4*(p_g4/k_g4)**n_g4 + a_g4_g6*(p_g6/k_g6)**n_g6 + a_g4_g5*(p_g5/k_g5)**n_g5 + a_g4_g3*(p_g3/k_g3)**n_g3 + a_g4_g4_g6*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 + a_g4_g4_g5*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 + a_g4_g4_g3*(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 + a_g4_g6_g5*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 + a_g4_g6_g3*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g4_g5_g3*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g4_g4_g6_g5*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 + a_g4_g4_g6_g3*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g4_g4_g5_g3*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g4_g6_g5_g3*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g4_g4_g6_g5_g3*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 )/( 1 +(p_g4/k_g4)**n_g4 +(p_g6/k_g6)**n_g6 +(p_g5/k_g5)**n_g5 +(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 +(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 +(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 ))-l_x_g4*x_g4
    dp_g4 = r_g4*x_g4- l_p_g4*p_g4
    dx_g1 = m_g1*(( alpha_g1 + a_g1_g4*(p_g4/k_g4)**n_g4 + a_g1_g5*(p_g5/k_g5)**n_g5 + a_g1_g6*(p_g6/k_g6)**n_g6 + a_g1_g4_g5*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 + a_g1_g4_g6*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 + a_g1_g5_g6*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 + a_g1_g4_g5_g6*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 )/( 1 +(p_g4/k_g4)**n_g4 +(p_g5/k_g5)**n_g5 +(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 +(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 ))-l_x_g1*x_g1
    dp_g1 = r_g1*x_g1- l_p_g1*p_g1
    dx_g5 = m_g5*(( alpha_g5 + a_g5_g4*(p_g4/k_g4)**n_g4 + a_g5_g6*(p_g6/k_g6)**n_g6 + a_g5_g5*(p_g5/k_g5)**n_g5 + a_g5_g3*(p_g3/k_g3)**n_g3 + a_g5_g4_g6*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 + a_g5_g4_g5*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 + a_g5_g4_g3*(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 + a_g5_g6_g5*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 + a_g5_g6_g3*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g5_g5_g3*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g5_g4_g6_g5*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 + a_g5_g4_g6_g3*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g5_g4_g5_g3*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g5_g6_g5_g3*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g5_g4_g6_g5_g3*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 )/( 1 +(p_g4/k_g4)**n_g4 +(p_g6/k_g6)**n_g6 +(p_g5/k_g5)**n_g5 +(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 +(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 +(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 ))-l_x_g5*x_g5
    dp_g5 = r_g5*x_g5- l_p_g5*p_g5
    dx_g6 = m_g6*(( alpha_g6 + a_g6_g4*(p_g4/k_g4)**n_g4 + a_g6_g5*(p_g5/k_g5)**n_g5 + a_g6_g6*(p_g6/k_g6)**n_g6 + a_g6_g3*(p_g3/k_g3)**n_g3 + a_g6_g7*(p_g7/k_g7)**n_g7 + a_g6_g4_g5*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 + a_g6_g4_g6*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 + a_g6_g4_g3*(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 + a_g6_g4_g7*(p_g4/k_g4)**n_g4*(p_g7/k_g7)**n_g7 + a_g6_g5_g6*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 + a_g6_g5_g3*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g6_g5_g7*(p_g5/k_g5)**n_g5*(p_g7/k_g7)**n_g7 + a_g6_g6_g3*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g6_g6_g7*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 + a_g6_g3_g7*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g4_g5_g6*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 + a_g6_g4_g5_g3*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 + a_g6_g4_g5_g7*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g7/k_g7)**n_g7 + a_g6_g4_g6_g3*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g6_g4_g6_g7*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 + a_g6_g4_g3_g7*(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g5_g6_g3*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g6_g5_g6_g7*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 + a_g6_g5_g3_g7*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g6_g3_g7*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g4_g5_g6_g3*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 + a_g6_g4_g5_g6_g7*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 + a_g6_g4_g5_g3_g7*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g4_g6_g3_g7*(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g5_g6_g3_g7*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 + a_g6_g4_g5_g6_g3_g7*(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 )/( 1 +(p_g4/k_g4)**n_g4 +(p_g5/k_g5)**n_g5 +(p_g6/k_g6)**n_g6 +(p_g3/k_g3)**n_g3 +(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g7/k_g7)**n_g7 +(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 +(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g5/k_g5)**n_g5*(p_g7/k_g7)**n_g7 +(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 +(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 +(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 +(p_g4/k_g4)**n_g4*(p_g5/k_g5)**n_g5*(p_g6/k_g6)**n_g6*(p_g3/k_g3)**n_g3*(p_g7/k_g7)**n_g7 ))-l_x_g6*x_g6
    dp_g6 = r_g6*x_g6- l_p_g6*p_g6
    dx_g7 = m_g7*(( alpha_g7 + a_g7_g6*(p_g6/k_g6)**n_g6 )/( 1 +(p_g6/k_g6)**n_g6 ))-l_x_g7*x_g7
    dp_g7 = r_g7*x_g7- l_p_g7*p_g7
    dx_g8 = m_g8*(( alpha_g8 + a_g8_g8*(p_g8/k_g8)**n_g8 )/( 1 +(p_g8/k_g8)**n_g8 ))-l_x_g8*x_g8
    dp_g8 = r_g8*x_g8- l_p_g8*p_g8
    dY = np.array([dx_g2,dp_g2,dx_g3,dp_g3,dx_g4,dp_g4,dx_g1,dp_g1,dx_g5,dp_g5,dx_g6,dp_g6,dx_g7,dp_g7,dx_g8,dp_g8,])
    return(dY)
#####################################################