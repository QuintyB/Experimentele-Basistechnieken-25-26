import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# ══════════════════════════════════════════════════════════════════════
# VASTE PARAMETERS
# ══════════════════════════════════════════════════════════════════════
M1 = 0.069
M2 = 0.020
G  = 9.81
l1 = 0.167
l2 = 0.167

SIGMA_M       = 0.001
SIGMA_L_FIXED = 0.002
SIGMA_ORIGIN  = 0.003
SIGMA_CLICK   = 0.003

#BASE = '/Users/swinn/Documents/2de bachelor/experimentel basistechnieken/VS code/vrij proef'

# ══════════════════════════════════════════════════════════════════════
# DIVERGENTIE-INSTELLINGEN
# ══════════════════════════════════════════════════════════════════════
# Vier vaste absolute drempelwaarden [°]
DREMPEL_HEEL_KLEIN = 1.88   # representatieve meetfout op de hoek
DREMPEL_KLEIN      = 10.0   # kleine drempel
DREMPEL_GROOT      = 50.0   # grote drempel
DREMPEL_HEEL_GROOT = 90.0   # letterlijk ander kwadrant

N_AANEENGESLOTEN   = 3      # opeenvolgende frames boven drempel

# ── Ratio-divergentie ────────────────────────────────────────────────
# ratio(t) = |θ₂_exp(t) − θ₂_sim(t)| / max(|θ₂_sim|)
# Divergentietijd = eerste t waarbij ratio > ALPHA gedurende N_AANEENGESLOTEN frames.
# ALPHA = 0.05 komt overeen met een 5 %-afwijking t.o.v. de Lagrangiaanse amplitude.
ALPHA_RATIO = 0.05

# ══════════════════════════════════════════════════════════════════════
# RESULTATEN-OPSLAG
# ══════════════════════════════════════════════════════════════════════
resultaten = []   # lijst van dicts, één per proef

# ══════════════════════════════════════════════════════════════════════
# HULPFUNCTIES
# ══════════════════════════════════════════════════════════════════════
def read_tracker(path, ncols):
    rows = []
    with open(path, encoding='utf-8') as fh:
        for k, line in enumerate(fh):
            if k < 2:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', '.').split('\t')
            try:
                rows.append([float(parts[j]) for j in range(ncols)])
            except (ValueError, IndexError):
                pass
    return np.array(rows)

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def eom(_, state):
    th1, th2, w1, w2 = state
    delta = th1 - th2
    cd, sd = np.cos(delta), np.sin(delta)
    M11 = (M1 + M2) * L1**2
    M12 = M2 * L1 * L2 * cd
    M22 = M2 * L2**2
    f1  = -M2 * L1 * L2 * sd * w2**2 - (M1 + M2) * G * L1 * np.sin(th1)
    f2  = +M2 * L1 * L2 * sd * w1**2 - M2 * G * L2 * np.sin(th2)
    det  = M11 * M22 - M12**2
    alp1 = (M22 * f1 - M12 * f2) / det
    alp2 = (M11 * f2 - M12 * f1) / det
    return [w1, w2, alp1, alp2]

def energie(th1, th2, w1, w2, LL1, LL2):
    T = (0.5 * M1 * LL1**2 * w1**2
       + 0.5 * M2 * (LL1**2 * w1**2 + LL2**2 * w2**2
                     + 2 * LL1 * LL2 * w1 * w2 * np.cos(th1 - th2)))
    V = (-M1 * G * LL1 * np.cos(th1)
         - M2 * G * (LL1 * np.cos(th1) + LL2 * np.cos(th2)))
    return T, V, T + V

def sigma_energie(th1, th2, w1, w2, LL1, LL2,
                  s_th1, s_th2, s_w1, s_w2, s_L1, s_L2):
    bijdragen_T2 = np.zeros(len(th1))
    bijdragen_V2 = np.zeros(len(th1))
    bijdragen_E2 = np.zeros(len(th1))

    def voeg_toe(d_th1=0, d_th2=0, d_w1=0, d_w2=0, d_L1=0, d_L2=0):
        Tp, Vp, Ep = energie(th1+d_th1, th2+d_th2, w1+d_w1, w2+d_w2, LL1+d_L1, LL2+d_L2)
        Tm, Vm, Em = energie(th1-d_th1, th2-d_th2, w1-d_w1, w2-d_w2, LL1-d_L1, LL2-d_L2)
        bijdragen_T2[:] += ((Tp - Tm) / 2)**2
        bijdragen_V2[:] += ((Vp - Vm) / 2)**2
        bijdragen_E2[:] += ((Ep - Em) / 2)**2

    voeg_toe(d_th1=s_th1);  voeg_toe(d_th2=s_th2)
    voeg_toe(d_w1=s_w1);    voeg_toe(d_w2=s_w2)
    voeg_toe(d_L1=s_L1);    voeg_toe(d_L2=s_L2)
    return np.sqrt(bijdragen_T2), np.sqrt(bijdragen_V2), np.sqrt(bijdragen_E2)

# ══════════════════════════════════════════════════════════════════════
# DIVERGENTIEDETECTIE – ABSOLUTE DREMPEL
# ══════════════════════════════════════════════════════════════════════
def vind_divergentietijd(t, th2_exp_deg, th2_sim_deg, drempel, n_frames=N_AANEENGESLOTEN):
    """
    Eerste tijdstip waarop |θ₂_exp − θ₂_sim| gedurende minstens n_frames
    opeenvolgende frames > drempel [°].  Geeft None terug indien niet gevonden.
    """
    verschil = np.abs(th2_exp_deg - th2_sim_deg)
    boven    = verschil > drempel
    count = 0
    for idx in range(len(t)):
        if boven[idx]:
            count += 1
            if count >= n_frames:
                return t[idx - n_frames + 1]
        else:
            count = 0
    return None

# ══════════════════════════════════════════════════════════════════════
# DIVERGENTIEDETECTIE – GENORMALISEERDE RATIO
# ══════════════════════════════════════════════════════════════════════
def vind_ratio_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                               alpha=ALPHA_RATIO, n_frames=N_AANEENGESLOTEN):
    """
    Berekent de genormaliseerde afwijking:
        ratio(t) = |θ₂_exp(t) − θ₂_sim(t)| / max(|θ₂_sim|)

    Geeft het eerste tijdstip terug waarop ratio(t) > alpha gedurende
    minstens n_frames opeenvolgende frames.

    Normalisatie op max(|θ₂_sim|) maakt de drempel schaalinvariant:
    een 5 %-afwijking t.o.v. de simulatie-amplitude, ongeacht of de
    slinger 20° of 180° zwaait.
    """
    norm = np.max(np.abs(th2_sim_deg))
    if norm < 1e-6:          # simulatie staat stil: ratio ongedefinieerd
        return None
    ratio = np.abs(th2_exp_deg - th2_sim_deg) / norm
    boven = ratio > alpha
    count = 0
    for idx in range(len(t)):
        if boven[idx]:
            count += 1
            if count >= n_frames:
                return t[idx - n_frames + 1]
        else:
            count = 0
    return None

# ══════════════════════════════════════════════════════════════════════
# LOSLAAT-DETECTIE
# ══════════════════════════════════════════════════════════════════════
def vind_loslaat_index(w1, w2, n_rust=10, n_bevestig=5, factor=3.0):
    drempel_w1 = factor * np.std(w1[:n_rust]) + 1e-6
    drempel_w2 = factor * np.std(w2[:n_rust]) + 1e-6
    count = 0
    for idx in range(n_rust, len(w1)):
        if abs(w1[idx]) > drempel_w1 and abs(w2[idx]) > drempel_w2:
            count += 1
            if count >= n_bevestig:
                return idx - n_bevestig + 1
        else:
            count = 0
    print("    [!] Geen loslaat-moment gevonden, fallback naar frame 0")
    return 0

# ══════════════════════════════════════════════════════════════════════
# HOOFDLUS
# ══════════════════════════════════════════════════════════════════════
for i in range(1, 33):
    FILE_A  = f'Vrije proef/data/proef_2_{i}_A.txt'
    FILE_B  = f'Vrije proef/data/proef_2_{i}_B.txt'
    OUT_PNG = f'Vrije proef/fotos/dubbele_slinger_{i}_2_0.png'

    raw_B = read_tracker(FILE_B, 3)
    raw_A = read_tracker(FILE_A, 3)
    N_frames = min(len(raw_B), len(raw_A))
    raw_B, raw_A = raw_B[:N_frames], raw_A[:N_frames]

    t  = raw_B[:, 0] / 2
    dt = np.mean(np.diff(t))

    xB = raw_B[:, 1];  yB = raw_B[:, 2]
    xA = raw_A[:, 1];  yA = raw_A[:, 2]

    afstanden_B     = np.sqrt(xB**2 + yB**2)
    L1              = np.mean(afstanden_B)
    sigma_tracker_B = np.std(afstanden_B)

    afstanden_A     = np.sqrt((xA - xB)**2 + (yA - yB)**2)
    L2              = np.mean(afstanden_A)
    sigma_tracker_A = np.std(afstanden_A)

    sigma_pos_B = np.sqrt(sigma_tracker_B**2 + SIGMA_CLICK**2 + SIGMA_ORIGIN**2)
    sigma_pos_B = max(sigma_pos_B, SIGMA_L_FIXED)
    sigma_pos_A = np.sqrt(sigma_tracker_A**2 + sigma_tracker_B**2
                          + 2 * SIGMA_CLICK**2 + SIGMA_ORIGIN**2)
    sigma_pos_A = max(sigma_pos_A, SIGMA_L_FIXED)
    sigma_L1 = sigma_pos_B
    sigma_L2 = sigma_pos_A

    th1_cont = np.unwrap(np.arctan2(xB, -yB))
    th2_cont = np.unwrap(np.arctan2(xA - xB, yB - yA))
    th_1     = wrap_angle(th1_cont)
    th_2     = wrap_angle(th2_cont)

    sigma_th1 = sigma_pos_B / L1
    sigma_th2 = sigma_pos_A / L2

    sigma_th1_zonder_origin = np.sqrt(sigma_tracker_B**2 + SIGMA_CLICK**2) / L1
    sigma_th2_zonder_origin = np.sqrt(sigma_tracker_A**2 + sigma_tracker_B**2
                                      + 2 * SIGMA_CLICK**2) / L2

    w1_exp = np.gradient(th1_cont, t)
    w2_exp = np.gradient(th2_cont, t)

    i_start = vind_loslaat_index(w1_exp, w2_exp, n_rust=10, n_bevestig=5, factor=3.0)
    t_start = t[i_start]

    t        = t       [i_start:]
    th_1     = th_1    [i_start:]
    th_2     = th_2    [i_start:]
    th1_cont = th1_cont[i_start:]
    th2_cont = th2_cont[i_start:]
    w1_exp   = w1_exp  [i_start:]
    w2_exp   = w2_exp  [i_start:]

    print(f"  Loslaat-frame: {i_start}  (t = {t_start:.3f} s ingeknipt)")

    th1_0 = th_1[0];   th2_0 = th_2[0]
    w1_0  = w1_exp[0]; w2_0  = w2_exp[0]

    sigma_w1 = sigma_th1_zonder_origin / (np.sqrt(2) * dt)
    sigma_w2 = sigma_th2_zonder_origin / (np.sqrt(2) * dt)

    sol = solve_ivp(eom,
        t_span=(t[0], t[-1]),
        y0=[th1_0, th2_0, w1_0, w2_0],
        method='RK45', t_eval=t,
        rtol=1e-9, atol=1e-11)

    th1_s = wrap_angle(sol.y[0]);  th2_s = wrap_angle(sol.y[1])
    w1_s  = sol.y[2];              w2_s  = sol.y[3]

    T_e, V_e, E_e = energie(th_1, th_2, w1_exp, w2_exp, L1, L2)
    T_s, V_s, E_s = energie(th1_s, th2_s, w1_s, w2_s, l1, l2)
    E0 = E_s[0]

    sT_e, sV_e, sE_e = sigma_energie(
        th_1, th_2, w1_exp, w2_exp, L1, L2,
        sigma_th1, sigma_th2, sigma_w1, sigma_w2, sigma_L1, sigma_L2)

    # ──────────────────────────────────────────────────────────────────
    # DIVERGENTIETIJDEN – 4 ABSOLUTE DREMPELS + 1 RATIO-DREMPEL
    # ──────────────────────────────────────────────────────────────────
    th2_exp_deg = np.degrees(th_2)
    th2_sim_deg = np.degrees(th2_s)

    t_div_heel_klein = vind_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                                            drempel=DREMPEL_HEEL_KLEIN)
    t_div_klein      = vind_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                                            drempel=DREMPEL_KLEIN)
    t_div_groot      = vind_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                                            drempel=DREMPEL_GROOT)
    t_div_heel_groot = vind_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                                            drempel=DREMPEL_HEEL_GROOT)
    t_div_ratio      = vind_ratio_divergentietijd(t, th2_exp_deg, th2_sim_deg,
                                                  alpha=ALPHA_RATIO)

    resultaten.append({
        'proef':             i,
        't_div_heel_klein':  t_div_heel_klein,
        't_div_klein':       t_div_klein,
        't_div_groot':       t_div_groot,
        't_div_heel_groot':  t_div_heel_groot,
        't_div_ratio':       t_div_ratio,
        't_meting':          t[-1],
    })

    print(f"  Proef {i:2d}  "
          f"t_div(1.88°)={t_div_heel_klein}  "
          f"t_div(10°)={t_div_klein}  "
          f"t_div(50°)={t_div_groot}  "
          f"t_div(90°)={t_div_heel_groot}  "
          f"t_div(ratio α=0.05)={t_div_ratio}")

    # ──────────────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────────────
    C_EXP = '#1f77b4'
    C_SIM = '#d62728'

    plt.rcParams.update({
        'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 17,
        'xtick.labelsize': 15, 'ytick.labelsize': 15,
        'legend.fontsize': 13.5, 'figure.dpi': 120, 'lines.linewidth': 1.3,
    })

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 1, hspace=0.65,
                            left=0.08, right=0.95, top=0.84, bottom=0.08)

    def decor(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel);  ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.35), framealpha=0.85)

    band_th2 = np.full_like(t, np.degrees(sigma_th2))

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.fill_between(t, th2_exp_deg - band_th2, th2_exp_deg + band_th2,
                        color=C_EXP, alpha=0.25, label='measurement uncertainty')
    ax_top.plot(t, th2_exp_deg, color=C_EXP, lw=1.2, label='Experiment')
    ax_top.plot(t, th2_sim_deg, color=C_SIM, lw=1.2, ls='--', label='Lagrangian')

#    if t_div_heel_klein is not None:
#        ax_top.axvline(t_div_heel_klein, color='orange',  ls=':', lw=1.5,
#                       label=f'div. 1.88°')
#    if t_div_klein is not None:
#        ax_top.axvline(t_div_klein,      color='green',   ls=':', lw=1.5,
#                       label=f'div. 10°')
#    if t_div_groot is not None:
#        ax_top.axvline(t_div_groot,      color='purple',  ls='--', lw=1.5,
#                       label=f'div. 50°')
#    if t_div_heel_groot is not None:
#        ax_top.axvline(t_div_heel_groot, color='black',   ls='--', lw=1.5,
#                       label=f'div. 90°')
#    if t_div_ratio is not None:
#        ax_top.axvline(t_div_ratio,      color='#e377c2', ls='-.', lw=1.5,
#                       label=f'div. ratio α=0.05')

    decor(ax_top, 'time  [s]', r'$\theta_2$  [$^\circ$]',
          'Angle of the second mass')

    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.fill_between(t, T_e-sT_e, T_e+sT_e, color='#ff7f0e', alpha=0.25)
    ax_bottom.fill_between(t, V_e-sV_e, V_e+sV_e, color='#2ca02c', alpha=0.25)
    ax_bottom.fill_between(t, E_e-sE_e, E_e+sE_e, color='#1f77b4', alpha=0.25)
    ax_bottom.plot(t, T_e, color='#ff7f0e', lw=1.2, label='Kinetic energy T')
    ax_bottom.plot(t, V_e, color='#2ca02c', lw=1.2, label='Potential energy V')
    ax_bottom.plot(t, E_e, color='#1f77b4', lw=2.2, label='Total energy E')
    ax_bottom.axhline(E0, color='red', label=f'$E_0$ = {E0:.3f} J  (theoretical)')
    decor(ax_bottom, 'time  [s]', 'Energy  [J]', 'Total energy')

    fig.suptitle(
        f'Double pendulum – Lagrangian vs experiment  (experiment {i})',
        fontsize=18, y=0.98)

    plt.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# SAMENVATTING – GEMIDDELDE DIVERGENTIETIJDEN MET FOUT
# ══════════════════════════════════════════════════════════════════════

def stats(key):
    """
    Geeft voor een gegeven sleutel de waarden terug waar divergentie
    wél gevonden werd, samen met:
      - gemiddelde  μ
      - standaarddeviatie op de populatie  σ
      - standaardfout op het gemiddelde    SEM = σ / √n
      - aantal gevonden proeven  n
      - aantal niet gevonden     n_niet
    """
    vals = np.array([r[key] for r in resultaten if r[key] is not None], dtype=float)
    n       = len(vals)
    n_niet  = len(resultaten) - n
    if n == 0:
        return None, None, None, 0, len(resultaten)
    mu  = np.mean(vals)
    sig = np.std(vals, ddof=1)      # steekproef-std
    sem = sig / np.sqrt(n)
    return mu, sig, sem, n, n_niet

def fmt_stats(key, label):
    mu, sig, sem, n, n_niet = stats(key)
    if mu is None:
        print(f"  {label:<30s}  –  (nooit gevonden in {n_niet} proeven)")
        return
    print(f"  {label:<30s}  "
          f"μ = {mu:.3f} s   "
          f"σ = {sig:.3f} s   "
          f"SEM = {sem:.3f} s   "
          f"(n = {n}/32, niet gevonden: {n_niet})")

print("\n" + "═" * 80)
print("  SAMENVATTING DIVERGENTIETIJDEN  (alle 32 proeven samen, geen chaos/regulier)")
print("═" * 80)
print(f"  N_aaneengesloten frames : {N_AANEENGESLOTEN}")
print(f"  Ratio-drempel α         : {ALPHA_RATIO}  "
      f"(= |θ₂_exp − θ₂_sim| / max|θ₂_sim| > {ALPHA_RATIO})")
print()
print(f"  {'Criterium':<30s}  {'Gemiddelde':^12s}   "
      f"{'Std':^10s}   {'SEM':^10s}   Gevonden")
print(f"  {'-'*30}  {'-'*12}   {'-'*10}   {'-'*10}   {'-'*10}")
fmt_stats('t_div_heel_klein',  'Absoluut  1.88° (meetfout)')
fmt_stats('t_div_klein',       'Absoluut 10°')
fmt_stats('t_div_groot',       'Absoluut 50°')
fmt_stats('t_div_heel_groot',  'Absoluut 90°')
fmt_stats('t_div_ratio',       f'Ratio α = {ALPHA_RATIO}')
print("═" * 80)