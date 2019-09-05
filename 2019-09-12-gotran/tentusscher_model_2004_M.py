# Gotran generated code for the  "tentusscher_model_2004_M" model
from __future__ import division

def init_state_values(**values):
    """
    Initialize state values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Init values
    # Xr1=0, Xr2=1, Xs=0, m=0, h=0.75, j=0.75, d=0, f=1, fCa=1, s=1, r=0,
    # g=1, Cai=0.0002, Ca_SR=0.2, Nai=11.6, V=-86.2, Ki=138.3
    init_values = np.array([0, 1, 0, 0, 0.75, 0.75, 0, 1, 1, 1, 0, 1, 0.0002,\
        0.2, 11.6, -86.2, 138.3], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("Xr1",(0, Range())), ("Xr2",(1, Range())), ("Xs",(2,\
        Range())), ("m",(3, Range())), ("h",(4, Range())), ("j",(5,\
        Range())), ("d",(6, Range())), ("f",(7, Range())), ("fCa",(8,\
        Range())), ("s",(9, Range())), ("r",(10, Range())), ("g",(11,\
        Range())), ("Cai",(12, Range())), ("Ca_SR",(13, Range())),\
        ("Nai",(14, Range())), ("V",(15, Range())), ("Ki",(16, Range()))])

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Param values
    # P_kna=0.03, lambda_CaL=0.0, lambda_K1=0.0, lambda_Kr=0.0,
    # lambda_Ks=0.0, lambda_Na=0.0, lambda_NaCa=0.0, lambda_NaK=0.0,
    # lambda_b_Ca=0.0, lambda_b_Na=0.0, lambda_leak=0.0, lambda_p_Ca=0.0,
    # lambda_p_K=0.0, lambda_rel=0.0, lambda_to=0.0, lambda_up=0.0,
    # g_K1=5.405, g_Kr=0.096, g_Ks=0.062, g_Na=14.838,
    # perc_reduced_inact_for_IpNa=0, shift_INa_inact=0.0, g_bna=0.00029,
    # g_CaL=0.000175, g_bca=0.000592, g_to=0.294, K_mNa=40.0, K_mk=1.0,
    # P_NaK=1.362, K_NaCa=1000.0, K_sat=0.1, Km_Ca=1.38, Km_Nai=87.5,
    # alpha=2.5, gamma=0.35, K_pCa=0.0005, g_pCa=0.825, g_pK=0.0146,
    # Buf_c=0.15, Buf_sr=10.0, Cao=2.0, K_buf_c=0.001, K_buf_sr=0.3,
    # K_up=0.00025, V_leak=8e-05, V_sr=0.001094, Vmax_up=0.000425,
    # a_rel=0.016464, b_rel=0.25, c_rel=0.008232, tau_g=2.0, Nao=140.0,
    # conc_clamp=1, Cm=0.185, F=96485.3415, R=8314.472, T=310.0,
    # Vmyo=0.016404, stim_amplitude=-52.0, stim_duration=1.0,
    # stim_end=5000.0, stim_period=2000.0, stim_start=100.0, Ko=5.4
    init_values = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,\
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.405, 0.096, 0.062, 14.838, 0,\
        0.0, 0.00029, 0.000175, 0.000592, 0.294, 40.0, 1.0, 1.362, 1000.0,\
        0.1, 1.38, 87.5, 2.5, 0.35, 0.0005, 0.825, 0.0146, 0.15, 10.0, 2.0,\
        0.001, 0.3, 0.00025, 8e-05, 0.001094, 0.000425, 0.016464, 0.25,\
        0.008232, 2.0, 140.0, 1, 0.185, 96485.3415, 8314.472, 310.0,\
        0.016404, -52.0, 1.0, 5000.0, 2000.0, 100.0, 5.4], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("P_kna", (0, Range())), ("lambda_CaL", (1, Range())),\
        ("lambda_K1", (2, Range())), ("lambda_Kr", (3, Range())),\
        ("lambda_Ks", (4, Range())), ("lambda_Na", (5, Range())),\
        ("lambda_NaCa", (6, Range())), ("lambda_NaK", (7, Range())),\
        ("lambda_b_Ca", (8, Range())), ("lambda_b_Na", (9, Range())),\
        ("lambda_leak", (10, Range())), ("lambda_p_Ca", (11, Range())),\
        ("lambda_p_K", (12, Range())), ("lambda_rel", (13, Range())),\
        ("lambda_to", (14, Range())), ("lambda_up", (15, Range())), ("g_K1",\
        (16, Range())), ("g_Kr", (17, Range())), ("g_Ks", (18, Range())),\
        ("g_Na", (19, Range())), ("perc_reduced_inact_for_IpNa", (20,\
        Range())), ("shift_INa_inact", (21, Range())), ("g_bna", (22,\
        Range())), ("g_CaL", (23, Range())), ("g_bca", (24, Range())),\
        ("g_to", (25, Range())), ("K_mNa", (26, Range())), ("K_mk", (27,\
        Range())), ("P_NaK", (28, Range())), ("K_NaCa", (29, Range())),\
        ("K_sat", (30, Range())), ("Km_Ca", (31, Range())), ("Km_Nai", (32,\
        Range())), ("alpha", (33, Range())), ("gamma", (34, Range())),\
        ("K_pCa", (35, Range())), ("g_pCa", (36, Range())), ("g_pK", (37,\
        Range())), ("Buf_c", (38, Range())), ("Buf_sr", (39, Range())),\
        ("Cao", (40, Range())), ("K_buf_c", (41, Range())), ("K_buf_sr", (42,\
        Range())), ("K_up", (43, Range())), ("V_leak", (44, Range())),\
        ("V_sr", (45, Range())), ("Vmax_up", (46, Range())), ("a_rel", (47,\
        Range())), ("b_rel", (48, Range())), ("c_rel", (49, Range())),\
        ("tau_g", (50, Range())), ("Nao", (51, Range())), ("conc_clamp", (52,\
        Range())), ("Cm", (53, Range())), ("F", (54, Range())), ("R", (55,\
        Range())), ("T", (56, Range())), ("Vmyo", (57, Range())),\
        ("stim_amplitude", (58, Range())), ("stim_duration", (59, Range())),\
        ("stim_end", (60, Range())), ("stim_period", (61, Range())),\
        ("stim_start", (62, Range())), ("Ko", (63, Range()))])

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("Xr1", 0), ("Xr2", 1), ("Xs", 2), ("m", 3), ("h", 4),\
        ("j", 5), ("d", 6), ("f", 7), ("fCa", 8), ("s", 9), ("r", 10), ("g",\
        11), ("Cai", 12), ("Ca_SR", 13), ("Nai", 14), ("V", 15), ("Ki", 16)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict([("P_kna", 0), ("lambda_CaL", 1), ("lambda_K1", 2),\
        ("lambda_Kr", 3), ("lambda_Ks", 4), ("lambda_Na", 5), ("lambda_NaCa",\
        6), ("lambda_NaK", 7), ("lambda_b_Ca", 8), ("lambda_b_Na", 9),\
        ("lambda_leak", 10), ("lambda_p_Ca", 11), ("lambda_p_K", 12),\
        ("lambda_rel", 13), ("lambda_to", 14), ("lambda_up", 15), ("g_K1",\
        16), ("g_Kr", 17), ("g_Ks", 18), ("g_Na", 19),\
        ("perc_reduced_inact_for_IpNa", 20), ("shift_INa_inact", 21),\
        ("g_bna", 22), ("g_CaL", 23), ("g_bca", 24), ("g_to", 25), ("K_mNa",\
        26), ("K_mk", 27), ("P_NaK", 28), ("K_NaCa", 29), ("K_sat", 30),\
        ("Km_Ca", 31), ("Km_Nai", 32), ("alpha", 33), ("gamma", 34),\
        ("K_pCa", 35), ("g_pCa", 36), ("g_pK", 37), ("Buf_c", 38), ("Buf_sr",\
        39), ("Cao", 40), ("K_buf_c", 41), ("K_buf_sr", 42), ("K_up", 43),\
        ("V_leak", 44), ("V_sr", 45), ("Vmax_up", 46), ("a_rel", 47),\
        ("b_rel", 48), ("c_rel", 49), ("tau_g", 50), ("Nao", 51),\
        ("conc_clamp", 52), ("Cm", 53), ("F", 54), ("R", 55), ("T", 56),\
        ("Vmyo", 57), ("stim_amplitude", 58), ("stim_duration", 59),\
        ("stim_end", 60), ("stim_period", 61), ("stim_start", 62), ("Ko",\
        63)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def monitor_indices(*monitored):
    """
    Monitor indices
    """
    monitor_inds = dict([("E_Na", 0), ("E_K", 1), ("E_Ks", 2), ("E_Ca", 3),\
        ("alpha_K1", 4), ("beta_K1", 5), ("xK1_inf", 6), ("i_K1", 7),\
        ("i_Kr", 8), ("xr1_inf", 9), ("alpha_xr1", 10), ("beta_xr1", 11),\
        ("tau_xr1", 12), ("xr2_inf", 13), ("alpha_xr2", 14), ("beta_xr2",\
        15), ("tau_xr2", 16), ("i_Ks", 17), ("xs_inf", 18), ("alpha_xs", 19),\
        ("beta_xs", 20), ("tau_xs", 21), ("i_Na", 22), ("m_inf", 23),\
        ("alpha_m", 24), ("beta_m", 25), ("tau_m", 26), ("h_inf", 27),\
        ("alpha_h", 28), ("beta_h", 29), ("tau_h", 30), ("j_inf", 31),\
        ("alpha_j", 32), ("beta_j", 33), ("tau_j", 34), ("i_b_Na", 35),\
        ("i_CaL", 36), ("d_inf", 37), ("alpha_d", 38), ("beta_d", 39),\
        ("gamma_d", 40), ("tau_d", 41), ("f_inf", 42), ("tau_f", 43),\
        ("alpha_fCa", 44), ("beta_fCa", 45), ("gama_fCa", 46), ("fCa_inf",\
        47), ("tau_fCa", 48), ("d_fCa", 49), ("i_b_Ca", 50), ("i_to", 51),\
        ("s_inf", 52), ("tau_s", 53), ("r_inf", 54), ("tau_r", 55), ("i_NaK",\
        56), ("i_NaCa", 57), ("i_p_Ca", 58), ("i_p_K", 59), ("i_rel", 60),\
        ("i_up", 61), ("i_leak", 62), ("g_inf", 63), ("d_g", 64),\
        ("Cai_bufc", 65), ("Ca_sr_bufsr", 66), ("i_Stim", 67), ("dXr1_dt",\
        68), ("dXr2_dt", 69), ("dXs_dt", 70), ("dm_dt", 71), ("dh_dt", 72),\
        ("dj_dt", 73), ("dd_dt", 74), ("df_dt", 75), ("dfCa_dt", 76),\
        ("ds_dt", 77), ("dr_dt", 78), ("dg_dt", 79), ("dCai_dt", 80),\
        ("dCa_SR_dt", 81), ("dNai_dt", 82), ("dV_dt", 83), ("dKi_dt", 84)])

    indices = []
    for monitor in monitored:
        if monitor not in monitor_inds:
            raise ValueError("Unknown monitored: '{0}'".format(monitor))
        indices.append(monitor_inds[monitor])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def rhs(states, t, parameters, values=None):
    """
    Compute the right hand side of the tentusscher_model_2004_M ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 17)
    Xr1, Xr2, Xs, m, h, j, d, f, fCa, s, r, g, Cai, Ca_SR, Nai, V, Ki = states

    # Assign parameters
    assert(len(parameters) == 64)
    P_kna=parameters[0]; lambda_CaL=parameters[1]; lambda_K1=parameters[2];\
        lambda_Kr=parameters[3]; lambda_Ks=parameters[4];\
        lambda_Na=parameters[5]; lambda_NaCa=parameters[6];\
        lambda_NaK=parameters[7]; lambda_b_Ca=parameters[8];\
        lambda_b_Na=parameters[9]; lambda_leak=parameters[10];\
        lambda_p_Ca=parameters[11]; lambda_p_K=parameters[12];\
        lambda_rel=parameters[13]; lambda_to=parameters[14];\
        lambda_up=parameters[15]; g_K1=parameters[16]; g_Kr=parameters[17];\
        g_Ks=parameters[18]; g_Na=parameters[19];\
        perc_reduced_inact_for_IpNa=parameters[20];\
        shift_INa_inact=parameters[21]; g_bna=parameters[22];\
        g_CaL=parameters[23]; g_bca=parameters[24]; g_to=parameters[25];\
        K_mNa=parameters[26]; K_mk=parameters[27]; P_NaK=parameters[28];\
        K_NaCa=parameters[29]; K_sat=parameters[30]; Km_Ca=parameters[31];\
        Km_Nai=parameters[32]; alpha=parameters[33]; gamma=parameters[34];\
        K_pCa=parameters[35]; g_pCa=parameters[36]; g_pK=parameters[37];\
        Buf_c=parameters[38]; Buf_sr=parameters[39]; Cao=parameters[40];\
        K_buf_c=parameters[41]; K_buf_sr=parameters[42]; K_up=parameters[43];\
        V_leak=parameters[44]; V_sr=parameters[45]; Vmax_up=parameters[46];\
        a_rel=parameters[47]; b_rel=parameters[48]; c_rel=parameters[49];\
        tau_g=parameters[50]; Nao=parameters[51]; Cm=parameters[53];\
        F=parameters[54]; R=parameters[55]; T=parameters[56];\
        Vmyo=parameters[57]; stim_amplitude=parameters[58];\
        stim_duration=parameters[59]; stim_period=parameters[61];\
        stim_start=parameters[62]; Ko=parameters[63]

    # Init return args
    if values is None:
        values = np.zeros((17,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (17,)

    # Expressions for the Reversal potentials component
    E_Na = R*T*math.log(Nao/Nai)/F
    E_K = R*T*math.log(Ko/Ki)/F
    E_Ks = R*T*math.log((Ko + Nao*P_kna)/(P_kna*Nai + Ki))/F
    E_Ca = 0.5*R*T*math.log(Cao/Cai)/F

    # Expressions for the Inward rectifier potassium current component
    alpha_K1 = 0.1/(1.0 + 6.14421235332821e-06*math.exp(0.06*V - 0.06*E_K))
    beta_K1 = (0.36787944117144233*math.exp(0.1*V - 0.1*E_K) +\
        3.0606040200802673*math.exp(0.0002*V - 0.0002*E_K))/(1.0 +\
        math.exp(0.5*E_K - 0.5*V))
    xK1_inf = alpha_K1/(alpha_K1 + beta_K1)
    i_K1 = 0.4303314829119352*g_K1*math.sqrt(Ko)*(1 + lambda_K1)*(-E_K +\
        V)*xK1_inf

    # Expressions for the Rapid time dependent potassium current component
    i_Kr = 0.4303314829119352*g_Kr*math.sqrt(Ko)*(1 + lambda_Kr)*(-E_K +\
        V)*Xr1*Xr2

    # Expressions for the Xr1 gate component
    xr1_inf = 1.0/(1.0 + 0.02437284407327961*math.exp(-0.14285714285714285*V))
    alpha_xr1 = 450.0/(1.0 + 0.011108996538242306*math.exp(-0.1*V))
    beta_xr1 = 6.0/(1.0 + 13.581324522578193*math.exp(0.08695652173913043*V))
    tau_xr1 = 1.0*alpha_xr1*beta_xr1
    values[0] = (-Xr1 + xr1_inf)/tau_xr1

    # Expressions for the Xr2 gate component
    xr2_inf = 1.0/(1.0 + 39.12128399815321*math.exp(0.041666666666666664*V))
    alpha_xr2 = 3.0/(1.0 + 0.049787068367863944*math.exp(-0.05*V))
    beta_xr2 = 1.12/(1.0 + 0.049787068367863944*math.exp(0.05*V))
    tau_xr2 = 1.0*alpha_xr2*beta_xr2
    values[1] = (-Xr2 + xr2_inf)/tau_xr2

    # Expressions for the Slow time dependent potassium current component
    i_Ks = g_Ks*math.pow(Xs, 2.0)*(1 + lambda_Ks)*(-E_Ks + V)

    # Expressions for the Xs gate component
    xs_inf = 1.0/(1.0 + 0.6996725373751304*math.exp(-0.07142857142857142*V))
    alpha_xs = 1100.0/math.sqrt(1.0 +\
        0.18887560283756186*math.exp(-0.16666666666666666*V))
    beta_xs = 1.0/(1.0 + 0.049787068367863944*math.exp(0.05*V))
    tau_xs = 1.0*alpha_xs*beta_xs
    values[2] = (-Xs + xs_inf)/tau_xs

    # Expressions for the Fast sodium current component
    i_Na = g_Na*math.pow(m, 3.0)*(1 + lambda_Na)*(-E_Na + V)*h*j

    # Expressions for the m gate component
    m_inf = 1.0*math.pow(1.0 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V), -2.0)
    alpha_m = 1.0/(1.0 + 6.14421235332821e-06*math.exp(-0.2*V))
    beta_m = 0.1/(1.0 + 1096.6331584284585*math.exp(0.2*V)) + 0.1/(1.0 +\
        0.7788007830714049*math.exp(0.005*V))
    tau_m = 1.0*alpha_m*beta_m
    values[3] = (-m + m_inf)/tau_m

    # Expressions for the h gate component
    h_inf = 0.01*perc_reduced_inact_for_IpNa + math.pow(1.0 +\
        15212.593285654404*math.exp(0.13458950201884254*V -\
        0.13458950201884254*shift_INa_inact), -2.0)*(1.0 -\
        0.01*perc_reduced_inact_for_IpNa)
    alpha_h = (4.4312679295805147e-07*math.exp(-0.14705882352941177*V) if V <\
        -40.0 else 0)
    beta_h = (310000.0*math.exp(0.3485*V) + 2.7*math.exp(0.079*V) if V <\
        -40.0 else 0.77/(0.13 +\
        0.049758141083938695*math.exp(-0.0900900900900901*V)))
    tau_h = 1.0/(alpha_h + beta_h)
    values[4] = (-h + h_inf)/tau_h

    # Expressions for the j gate component
    j_inf = 0.01*perc_reduced_inact_for_IpNa + math.pow(1.0 +\
        15212.593285654404*math.exp(0.13458950201884254*V -\
        0.13458950201884254*shift_INa_inact), -2.0)*(1.0 -\
        0.01*perc_reduced_inact_for_IpNa)
    alpha_j = (1.0*(37.78 + V)*(-25428.0*math.exp(0.2444*V) -\
        6.948e-06*math.exp(-0.04391*V))/(1.0 +\
        50262745825.95399*math.exp(0.311*V)) if V < -40.0 else 0)
    beta_j = (0.02424*math.exp(-0.01052*V)/(1.0 +\
        0.003960868339904256*math.exp(-0.1378*V)) if V < -40.0 else\
        0.6*math.exp(0.057*V)/(1.0 + 0.040762203978366204*math.exp(-0.1*V)))
    tau_j = 1.0/(alpha_j + beta_j)
    values[5] = (-j + j_inf)/tau_j

    # Expressions for the Sodium background current component
    i_b_Na = g_bna*(1 + lambda_b_Na)*(-E_Na + V)

    # Expressions for the L_type Ca current component
    i_CaL = 4.0*g_CaL*math.pow(F, 2.0)*(1 + lambda_CaL)*(-0.341*Cao +\
        Cai*math.exp(2.0*F*V/(R*T)))*V*d*f*fCa/(R*T*(-1.0 +\
        math.exp(2.0*F*V/(R*T))))

    # Expressions for the d gate component
    d_inf = 1.0/(1.0 + 0.513417119032592*math.exp(-0.13333333333333333*V))
    alpha_d = 0.25 + 1.4/(1.0 +\
        0.0677244716592409*math.exp(-0.07692307692307693*V))
    beta_d = 1.4/(1.0 + 2.718281828459045*math.exp(0.2*V))
    gamma_d = 1.0/(1.0 + 12.182493960703473*math.exp(-0.05*V))
    tau_d = 1.0*alpha_d*beta_d + gamma_d
    values[6] = (-d + d_inf)/tau_d

    # Expressions for the f gate component
    f_inf = 1.0/(1.0 + 17.411708063327644*math.exp(0.14285714285714285*V))
    tau_f = 80.0 + 165.0/(1.0 + 12.182493960703473*math.exp(-0.1*V)) +\
        1125.0*math.exp(-0.004166666666666667*math.pow(27.0 + V, 2.0))
    values[7] = (-f + f_inf)/tau_f

    # Expressions for the FCa gate component
    alpha_fCa = 1.0/(1.0 + 8.03402376701711e+27*math.pow(Cai, 8.0))
    beta_fCa = 0.1/(1.0 + 0.006737946999085467*math.exp(10000.0*Cai))
    gama_fCa = 0.2/(1.0 + 0.391605626676799*math.exp(1250.0*Cai))
    fCa_inf = 0.15753424657534246 + 0.684931506849315*alpha_fCa +\
        0.684931506849315*beta_fCa + 0.684931506849315*gama_fCa
    tau_fCa = 2.0
    d_fCa = (-fCa + fCa_inf)/tau_fCa
    values[8] = (0 if fCa_inf > fCa and V > -60.0 else d_fCa)

    # Expressions for the Calcium background current component
    i_b_Ca = g_bca*(1 + lambda_b_Ca)*(-E_Ca + V)

    # Expressions for the Transient outward current component
    i_to = g_to*(1 + lambda_to)*(-E_K + V)*r*s

    # Expressions for the s gate component
    s_inf = 1.0/(1.0 + 54.598150033144236*math.exp(0.2*V))
    tau_s = 3.0 + 5.0/(1.0 + 0.01831563888873418*math.exp(0.2*V)) +\
        85.0*math.exp(-0.003125*math.pow(45.0 + V, 2.0))
    values[9] = (-s + s_inf)/tau_s

    # Expressions for the r gate component
    r_inf = 1.0/(1.0 + 28.031624894526125*math.exp(-0.16666666666666666*V))
    tau_r = 0.8 + 9.5*math.exp(-0.0005555555555555556*math.pow(40.0 + V, 2.0))
    values[10] = (-r + r_inf)/tau_r

    # Expressions for the Sodium potassium pump current component
    i_NaK = Ko*P_NaK*(1 + lambda_NaK)*Nai/((K_mNa + Nai)*(K_mk + Ko)*(1.0 +\
        0.0353*math.exp(-F*V/(R*T)) + 0.1245*math.exp(-0.1*F*V/(R*T))))

    # Expressions for the Sodium calcium exchanger current component
    i_NaCa = K_NaCa*(1 + lambda_NaCa)*(Cao*math.pow(Nai,\
        3.0)*math.exp(F*gamma*V/(R*T)) - alpha*math.pow(Nao,\
        3.0)*Cai*math.exp(F*(-1.0 + gamma)*V/(R*T)))/((1.0 +\
        K_sat*math.exp(F*(-1.0 + gamma)*V/(R*T)))*(Cao +\
        Km_Ca)*(math.pow(Km_Nai, 3.0) + math.pow(Nao, 3.0)))

    # Expressions for the Calcium pump current component
    i_p_Ca = g_pCa*(1 + lambda_p_Ca)*Cai/(K_pCa + Cai)

    # Expressions for the Potassium pump current component
    i_p_K = g_pK*(1 + lambda_p_K)*(-E_K + V)/(1.0 +\
        65.40521574193832*math.exp(-0.16722408026755853*V))

    # Expressions for the Calcium dynamics component
    i_rel = (1 + lambda_rel)*(c_rel + a_rel*math.pow(Ca_SR,\
        2.0)/(math.pow(b_rel, 2.0) + math.pow(Ca_SR, 2.0)))*d*g
    i_up = Vmax_up*(1 + lambda_up)/(1.0 + math.pow(K_up, 2.0)*math.pow(Cai,\
        -2.0))
    i_leak = V_leak*(1 + lambda_leak)*(-Cai + Ca_SR)
    g_inf = (1.0/(1.0 + 5.439910241481018e+20*math.pow(Cai, 6.0)) if Cai <\
        0.00035 else 1.0/(1.0 + 1.9720198874049195e+55*math.pow(Cai, 16.0)))
    d_g = (-g + g_inf)/tau_g
    values[11] = (0 if g_inf > g and V > -60.0 else d_g)
    Cai_bufc = 1.0/(1.0 + Buf_c*K_buf_c*math.pow(K_buf_c + Cai, -2.0))
    Ca_sr_bufsr = 1.0/(1.0 + Buf_sr*K_buf_sr*math.pow(K_buf_sr + Ca_SR, -2.0))
    values[12] = (-i_up - 0.5*Cm*(1.0*i_CaL + 1.0*i_b_Ca + 1.0*i_p_Ca -\
        2.0*i_NaCa)/(F*Vmyo) + i_leak + i_rel)*Cai_bufc
    values[13] = Vmyo*(-i_leak - i_rel + i_up)*Ca_sr_bufsr/V_sr

    # Expressions for the Sodium dynamics component
    values[14] = 0

    # Expressions for the Membrane component
    i_Stim = (stim_amplitude if t - stim_period*math.floor(t/stim_period) <=\
        stim_duration + stim_start and t -\
        stim_period*math.floor(t/stim_period) >= stim_start else 0)
    values[15] = -1.0*i_Stim - 1.0*(1 + lambda_CaL)*i_CaL - 1.0*(1 +\
        lambda_K1)*i_K1 - 1.0*(1 + lambda_Kr)*i_Kr - 1.0*(1 + lambda_Ks)*i_Ks\
        - 1.0*(1 + lambda_Na)*i_Na - 1.0*(1 + lambda_NaCa)*i_NaCa - 1.0*(1 +\
        lambda_NaK)*i_NaK - 1.0*(1 + lambda_b_Ca)*i_b_Ca - 1.0*(1 +\
        lambda_b_Na)*i_b_Na - 1.0*(1 + lambda_p_Ca)*i_p_Ca - 1.0*(1 +\
        lambda_p_K)*i_p_K - 1.0*(1 + lambda_to)*i_to

    # Expressions for the Potassium dynamics component
    values[16] = 0

    # Return results
    return values

def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the tentusscher_model_2004_M ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 17)
    Xr1, Xr2, Xs, m, h, j, d, f, fCa, s, r, g, Cai, Ca_SR, Nai, V, Ki = states

    # Assign parameters
    assert(len(parameters) == 64)
    P_kna=parameters[0]; lambda_CaL=parameters[1]; lambda_K1=parameters[2];\
        lambda_Kr=parameters[3]; lambda_Ks=parameters[4];\
        lambda_Na=parameters[5]; lambda_NaCa=parameters[6];\
        lambda_NaK=parameters[7]; lambda_b_Ca=parameters[8];\
        lambda_b_Na=parameters[9]; lambda_leak=parameters[10];\
        lambda_p_Ca=parameters[11]; lambda_p_K=parameters[12];\
        lambda_rel=parameters[13]; lambda_to=parameters[14];\
        lambda_up=parameters[15]; g_K1=parameters[16]; g_Kr=parameters[17];\
        g_Ks=parameters[18]; g_Na=parameters[19];\
        perc_reduced_inact_for_IpNa=parameters[20];\
        shift_INa_inact=parameters[21]; g_bna=parameters[22];\
        g_CaL=parameters[23]; g_bca=parameters[24]; g_to=parameters[25];\
        K_mNa=parameters[26]; K_mk=parameters[27]; P_NaK=parameters[28];\
        K_NaCa=parameters[29]; K_sat=parameters[30]; Km_Ca=parameters[31];\
        Km_Nai=parameters[32]; alpha=parameters[33]; gamma=parameters[34];\
        K_pCa=parameters[35]; g_pCa=parameters[36]; g_pK=parameters[37];\
        Buf_c=parameters[38]; Buf_sr=parameters[39]; Cao=parameters[40];\
        K_buf_c=parameters[41]; K_buf_sr=parameters[42]; K_up=parameters[43];\
        V_leak=parameters[44]; V_sr=parameters[45]; Vmax_up=parameters[46];\
        a_rel=parameters[47]; b_rel=parameters[48]; c_rel=parameters[49];\
        tau_g=parameters[50]; Nao=parameters[51]; Cm=parameters[53];\
        F=parameters[54]; R=parameters[55]; T=parameters[56];\
        Vmyo=parameters[57]; stim_amplitude=parameters[58];\
        stim_duration=parameters[59]; stim_period=parameters[61];\
        stim_start=parameters[62]; Ko=parameters[63]

    # Init return args
    if monitored is None:
        monitored = np.zeros((85,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (85,)

    # Expressions for the Reversal potentials component
    monitored[0] = R*T*math.log(Nao/Nai)/F
    monitored[1] = R*T*math.log(Ko/Ki)/F
    monitored[2] = R*T*math.log((Ko + Nao*P_kna)/(P_kna*Nai + Ki))/F
    monitored[3] = 0.5*R*T*math.log(Cao/Cai)/F

    # Expressions for the Inward rectifier potassium current component
    monitored[4] = 0.1/(1.0 + 6.14421235332821e-06*math.exp(0.06*V -\
        0.06*monitored[1]))
    monitored[5] = (0.36787944117144233*math.exp(0.1*V - 0.1*monitored[1]) +\
        3.0606040200802673*math.exp(0.0002*V - 0.0002*monitored[1]))/(1.0 +\
        math.exp(0.5*monitored[1] - 0.5*V))
    monitored[6] = monitored[4]/(monitored[4] + monitored[5])
    monitored[7] = 0.4303314829119352*g_K1*math.sqrt(Ko)*(1 +\
        lambda_K1)*(-monitored[1] + V)*monitored[6]

    # Expressions for the Rapid time dependent potassium current component
    monitored[8] = 0.4303314829119352*g_Kr*math.sqrt(Ko)*(1 +\
        lambda_Kr)*(-monitored[1] + V)*Xr1*Xr2

    # Expressions for the Xr1 gate component
    monitored[9] = 1.0/(1.0 +\
        0.02437284407327961*math.exp(-0.14285714285714285*V))
    monitored[10] = 450.0/(1.0 + 0.011108996538242306*math.exp(-0.1*V))
    monitored[11] = 6.0/(1.0 +\
        13.581324522578193*math.exp(0.08695652173913043*V))
    monitored[12] = 1.0*monitored[10]*monitored[11]
    monitored[68] = (-Xr1 + monitored[9])/monitored[12]

    # Expressions for the Xr2 gate component
    monitored[13] = 1.0/(1.0 +\
        39.12128399815321*math.exp(0.041666666666666664*V))
    monitored[14] = 3.0/(1.0 + 0.049787068367863944*math.exp(-0.05*V))
    monitored[15] = 1.12/(1.0 + 0.049787068367863944*math.exp(0.05*V))
    monitored[16] = 1.0*monitored[14]*monitored[15]
    monitored[69] = (-Xr2 + monitored[13])/monitored[16]

    # Expressions for the Slow time dependent potassium current component
    monitored[17] = g_Ks*math.pow(Xs, 2.0)*(1 + lambda_Ks)*(-monitored[2] + V)

    # Expressions for the Xs gate component
    monitored[18] = 1.0/(1.0 +\
        0.6996725373751304*math.exp(-0.07142857142857142*V))
    monitored[19] = 1100.0/math.sqrt(1.0 +\
        0.18887560283756186*math.exp(-0.16666666666666666*V))
    monitored[20] = 1.0/(1.0 + 0.049787068367863944*math.exp(0.05*V))
    monitored[21] = 1.0*monitored[19]*monitored[20]
    monitored[70] = (-Xs + monitored[18])/monitored[21]

    # Expressions for the Fast sodium current component
    monitored[22] = g_Na*math.pow(m, 3.0)*(1 + lambda_Na)*(-monitored[0] +\
        V)*h*j

    # Expressions for the m gate component
    monitored[23] = 1.0*math.pow(1.0 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V), -2.0)
    monitored[24] = 1.0/(1.0 + 6.14421235332821e-06*math.exp(-0.2*V))
    monitored[25] = 0.1/(1.0 + 1096.6331584284585*math.exp(0.2*V)) + 0.1/(1.0 +\
        0.7788007830714049*math.exp(0.005*V))
    monitored[26] = 1.0*monitored[24]*monitored[25]
    monitored[71] = (-m + monitored[23])/monitored[26]

    # Expressions for the h gate component
    monitored[27] = 0.01*perc_reduced_inact_for_IpNa + math.pow(1.0 +\
        15212.593285654404*math.exp(0.13458950201884254*V -\
        0.13458950201884254*shift_INa_inact), -2.0)*(1.0 -\
        0.01*perc_reduced_inact_for_IpNa)
    monitored[28] = (4.4312679295805147e-07*math.exp(-0.14705882352941177*V)\
        if V < -40.0 else 0)
    monitored[29] = (310000.0*math.exp(0.3485*V) + 2.7*math.exp(0.079*V) if V\
        < -40.0 else 0.77/(0.13 +\
        0.049758141083938695*math.exp(-0.0900900900900901*V)))
    monitored[30] = 1.0/(monitored[28] + monitored[29])
    monitored[72] = (-h + monitored[27])/monitored[30]

    # Expressions for the j gate component
    monitored[31] = 0.01*perc_reduced_inact_for_IpNa + math.pow(1.0 +\
        15212.593285654404*math.exp(0.13458950201884254*V -\
        0.13458950201884254*shift_INa_inact), -2.0)*(1.0 -\
        0.01*perc_reduced_inact_for_IpNa)
    monitored[32] = (1.0*(37.78 + V)*(-25428.0*math.exp(0.2444*V) -\
        6.948e-06*math.exp(-0.04391*V))/(1.0 +\
        50262745825.95399*math.exp(0.311*V)) if V < -40.0 else 0)
    monitored[33] = (0.02424*math.exp(-0.01052*V)/(1.0 +\
        0.003960868339904256*math.exp(-0.1378*V)) if V < -40.0 else\
        0.6*math.exp(0.057*V)/(1.0 + 0.040762203978366204*math.exp(-0.1*V)))
    monitored[34] = 1.0/(monitored[32] + monitored[33])
    monitored[73] = (-j + monitored[31])/monitored[34]

    # Expressions for the Sodium background current component
    monitored[35] = g_bna*(1 + lambda_b_Na)*(-monitored[0] + V)

    # Expressions for the L_type Ca current component
    monitored[36] = 4.0*g_CaL*math.pow(F, 2.0)*(1 + lambda_CaL)*(-0.341*Cao +\
        Cai*math.exp(2.0*F*V/(R*T)))*V*d*f*fCa/(R*T*(-1.0 +\
        math.exp(2.0*F*V/(R*T))))

    # Expressions for the d gate component
    monitored[37] = 1.0/(1.0 +\
        0.513417119032592*math.exp(-0.13333333333333333*V))
    monitored[38] = 0.25 + 1.4/(1.0 +\
        0.0677244716592409*math.exp(-0.07692307692307693*V))
    monitored[39] = 1.4/(1.0 + 2.718281828459045*math.exp(0.2*V))
    monitored[40] = 1.0/(1.0 + 12.182493960703473*math.exp(-0.05*V))
    monitored[41] = 1.0*monitored[38]*monitored[39] + monitored[40]
    monitored[74] = (-d + monitored[37])/monitored[41]

    # Expressions for the f gate component
    monitored[42] = 1.0/(1.0 +\
        17.411708063327644*math.exp(0.14285714285714285*V))
    monitored[43] = 80.0 + 165.0/(1.0 + 12.182493960703473*math.exp(-0.1*V))\
        + 1125.0*math.exp(-0.004166666666666667*math.pow(27.0 + V, 2.0))
    monitored[75] = (-f + monitored[42])/monitored[43]

    # Expressions for the FCa gate component
    monitored[44] = 1.0/(1.0 + 8.03402376701711e+27*math.pow(Cai, 8.0))
    monitored[45] = 0.1/(1.0 + 0.006737946999085467*math.exp(10000.0*Cai))
    monitored[46] = 0.2/(1.0 + 0.391605626676799*math.exp(1250.0*Cai))
    monitored[47] = 0.15753424657534246 + 0.684931506849315*monitored[44] +\
        0.684931506849315*monitored[45] + 0.684931506849315*monitored[46]
    monitored[48] = 2.0
    monitored[49] = (-fCa + monitored[47])/monitored[48]
    monitored[76] = (0 if monitored[47] > fCa and V > -60.0 else monitored[49])

    # Expressions for the Calcium background current component
    monitored[50] = g_bca*(1 + lambda_b_Ca)*(-monitored[3] + V)

    # Expressions for the Transient outward current component
    monitored[51] = g_to*(1 + lambda_to)*(-monitored[1] + V)*r*s

    # Expressions for the s gate component
    monitored[52] = 1.0/(1.0 + 54.598150033144236*math.exp(0.2*V))
    monitored[53] = 3.0 + 5.0/(1.0 + 0.01831563888873418*math.exp(0.2*V)) +\
        85.0*math.exp(-0.003125*math.pow(45.0 + V, 2.0))
    monitored[77] = (-s + monitored[52])/monitored[53]

    # Expressions for the r gate component
    monitored[54] = 1.0/(1.0 +\
        28.031624894526125*math.exp(-0.16666666666666666*V))
    monitored[55] = 0.8 + 9.5*math.exp(-0.0005555555555555556*math.pow(40.0 +\
        V, 2.0))
    monitored[78] = (-r + monitored[54])/monitored[55]

    # Expressions for the Sodium potassium pump current component
    monitored[56] = Ko*P_NaK*(1 + lambda_NaK)*Nai/((K_mNa + Nai)*(K_mk +\
        Ko)*(1.0 + 0.0353*math.exp(-F*V/(R*T)) +\
        0.1245*math.exp(-0.1*F*V/(R*T))))

    # Expressions for the Sodium calcium exchanger current component
    monitored[57] = K_NaCa*(1 + lambda_NaCa)*(Cao*math.pow(Nai,\
        3.0)*math.exp(F*gamma*V/(R*T)) - alpha*math.pow(Nao,\
        3.0)*Cai*math.exp(F*(-1.0 + gamma)*V/(R*T)))/((1.0 +\
        K_sat*math.exp(F*(-1.0 + gamma)*V/(R*T)))*(Cao +\
        Km_Ca)*(math.pow(Km_Nai, 3.0) + math.pow(Nao, 3.0)))

    # Expressions for the Calcium pump current component
    monitored[58] = g_pCa*(1 + lambda_p_Ca)*Cai/(K_pCa + Cai)

    # Expressions for the Potassium pump current component
    monitored[59] = g_pK*(1 + lambda_p_K)*(-monitored[1] + V)/(1.0 +\
        65.40521574193832*math.exp(-0.16722408026755853*V))

    # Expressions for the Calcium dynamics component
    monitored[60] = (1 + lambda_rel)*(c_rel + a_rel*math.pow(Ca_SR,\
        2.0)/(math.pow(b_rel, 2.0) + math.pow(Ca_SR, 2.0)))*d*g
    monitored[61] = Vmax_up*(1 + lambda_up)/(1.0 + math.pow(K_up,\
        2.0)*math.pow(Cai, -2.0))
    monitored[62] = V_leak*(1 + lambda_leak)*(-Cai + Ca_SR)
    monitored[63] = (1.0/(1.0 + 5.439910241481018e+20*math.pow(Cai, 6.0)) if\
        Cai < 0.00035 else 1.0/(1.0 + 1.9720198874049195e+55*math.pow(Cai,\
        16.0)))
    monitored[64] = (-g + monitored[63])/tau_g
    monitored[79] = (0 if monitored[63] > g and V > -60.0 else monitored[64])
    monitored[65] = 1.0/(1.0 + Buf_c*K_buf_c*math.pow(K_buf_c + Cai, -2.0))
    monitored[66] = 1.0/(1.0 + Buf_sr*K_buf_sr*math.pow(K_buf_sr + Ca_SR,\
        -2.0))
    monitored[80] = (-monitored[61] - 0.5*Cm*(1.0*monitored[36] +\
        1.0*monitored[50] + 1.0*monitored[58] - 2.0*monitored[57])/(F*Vmyo) +\
        monitored[60] + monitored[62])*monitored[65]
    monitored[81] = Vmyo*(-monitored[60] - monitored[62] +\
        monitored[61])*monitored[66]/V_sr

    # Expressions for the Sodium dynamics component
    monitored[82] = 0

    # Expressions for the Membrane component
    monitored[67] = (stim_amplitude if t -\
        stim_period*math.floor(t/stim_period) <= stim_duration + stim_start\
        and t - stim_period*math.floor(t/stim_period) >= stim_start else 0)
    monitored[83] = -1.0*monitored[67] - 1.0*(1 + lambda_CaL)*monitored[36] -\
        1.0*(1 + lambda_K1)*monitored[7] - 1.0*(1 + lambda_Kr)*monitored[8] -\
        1.0*(1 + lambda_Ks)*monitored[17] - 1.0*(1 + lambda_Na)*monitored[22]\
        - 1.0*(1 + lambda_NaCa)*monitored[57] - 1.0*(1 +\
        lambda_NaK)*monitored[56] - 1.0*(1 + lambda_b_Ca)*monitored[50] -\
        1.0*(1 + lambda_b_Na)*monitored[35] - 1.0*(1 +\
        lambda_p_Ca)*monitored[58] - 1.0*(1 + lambda_p_K)*monitored[59] -\
        1.0*(1 + lambda_to)*monitored[51]

    # Expressions for the Potassium dynamics component
    monitored[84] = 0

    # Return results
    return monitored
