#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

#parameters 
M1 = 0.069   # kg 
M2 = 0.020  # kg 
G  = 9.81    # m/s²

#data en output, aanpassen per file
FILE_B = 'Vrije proef/data/proef_2_3_B.txt'
FILE_A = 'Vrije proef/data/proef_2_3_A.txt'
OUT_PNG = 'Vrije proef/fotos/dubbele_slinger.png'
#OUT_PDF = 'Vrije proef/dubbele_slinger.pdf'

# inlezen data

def read_tracker(path, ncols):
    """
    Leest een Tracker-exportbestand (tab-gescheiden, komma als decimaalteken).
    Slaat de eerste 2 regelheaders over.
    Geeft een (N, ncols) numpy-array terug.
    """
    rows = []
    with open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i < 2: 
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

raw_B = read_tracker(FILE_B, 3)   # t  x  y 
raw_A = read_tracker(FILE_A, 3)   # t  x  y

# Controleer gelijke lengte
N = min(len(raw_B), len(raw_A))
raw_B, raw_A = raw_B[:N], raw_A[:N]

t             = raw_B[:, 0] /2         # tijdas (s), delen door twee want slowmotion

# data toekennen

xB = raw_B[:, 1]
yB = raw_B[:, 2]
xA = raw_A[:, 1]
yA = raw_A[:, 2]

# lengte bepaling door gemiddelde te nemen

L1 = np.mean(np.sqrt(xB**2 + yB**2))
L2 = np.mean(np.sqrt((xA - xB)**2 + (yA - yB)**2))

print("══════════════════════════════════════════════")
print("  Geometrie (afgeleid uit data)")
print(f"  L₁            = {L1*100:.2f} cm")
print(f"  L₂            = {L2*100:.2f} cm")


# hulp functie
def wrap_angle(theta):
    """Zorgt ervoor dat de hoek altijd tussen -π en +π (-180° en 180°) blijft."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

# Continue hoeken berekenen (noodzakelijk voor correcte afgeleide/snelheid!)
th1_cont = np.unwrap(np.arctan2(xB, -yB))
th2_cont = np.unwrap(np.arctan2(xA - xB, yB - yA))

# Hoeksnelheden berekenen met de continue data
w1_exp = np.gradient(th1_cont, t)
w2_exp = np.gradient(th2_cont, t)

# Nu de hoeken voor weergave en plotten vastzetten tussen -180° en 180°
th_1 = wrap_angle(th1_cont)
th_2 = wrap_angle(th2_cont)

# Beginvoorwaarden (gebruik de gewrapte hoeken)
th1_0 = th_1[0]
th2_0 = th_2[0]
w1_0  = w1_exp[0]
w2_0  = w2_exp[0]

print(f"\n  Beginvoorwaarden")
print(f"  θ₁(0) = {np.degrees(th1_0):+7.2f}°   ω₁(0) = {np.degrees(w1_0):+7.2f} °/s")
print(f"  θ₂(0) = {np.degrees(th2_0):+7.2f}°   ω₂(0) = {np.degrees(w2_0):+7.2f} °/s")
print("══════════════════════════════════════════════\n")


#simulatie

def eom(_, state):
    """
    Bewegingsvergelijkingen afgeleid van de Lagrangiaan voor een
    dubbele slinger met TWEE PUNTMASSA'S (massaloze staven).
    Staat = [θ₁, θ₂, ω₁, ω₂].
    """
    th1, th2, w1, w2 = state
    delta = th1 - th2
    cd, sd = np.cos(delta), np.sin(delta)

    # Massamatrix voor puntmassa's
    M11 = (M1 + M2) * L1**2
    M12 = M2 * L1 * L2 * cd
    M22 = M2 * L2**2

    # Rechterlid (middelpuntzoekende krachten en zwaartekracht)
    f1 = -M2 * L1 * L2 * sd * w2**2 - (M1 + M2) * G * L1 * np.sin(th1)
    f2 = +M2 * L1 * L2 * sd * w1**2 - M2 * G * L2 * np.sin(th2)

    # Oplos M·α = f  (2×2, analytisch)
    det  = M11 * M22 - M12**2
    alp1 = (M22 * f1 - M12 * f2) / det
    alp2 = (M11 * f2 - M12 * f1) / det

    return [w1, w2, alp1, alp2]

print("  Numerieke integratie (RK45) bezig …", end='', flush=True)
sol = solve_ivp(eom,
    t_span=(t[0], t[-1]),
    y0=[th1_0, th2_0, w1_0, w2_0],
    method='RK45',
    t_eval=t,
    rtol=1e-9,
    atol=1e-11,
    dense_output=False,
)
print(f"  klaar  (status={sol.status})")

# resultaten van simulatie 
# resultaten van simulatie (raw)
th1_s_raw = sol.y[0]
th2_s_raw = sol.y[1]
w1_s      = sol.y[2]
w2_s      = sol.y[3]

# Simulatiedata terugvouwen naar -180° tot 180°
th1_s = wrap_angle(th1_s_raw)
th2_s = wrap_angle(th2_s_raw)

# simulatie in x en y coordinaten

xB_s = L1*np.sin(th1_s)
yB_s = -L1*np.cos(th1_s)
xA_s = xB_s + L2*np.sin(th2_s)
yA_s = yB_s - L2*np.cos(th2_s)


# energie

def energie(th1, th2, w1, w2):

    # Kinetische energie # zie langrangiaan
    T = (0.5 * M1 * L1**2 * w1**2
         + 0.5 * M2 * (L1**2 * w1**2 + L2**2 * w2**2 + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)))
    
    # Potentiële energie
    V = (-M1 * G * L1 * np.cos(th1)
         - M2 * G * (L1 * np.cos(th1) + L2 * np.cos(th2)))
         
    return T, V, T + V

T_e, V_e, E_e = energie(th_1, th_2, w1_exp, w2_exp)
T_s, V_s, E_s = energie(th1_s,   th2_s,   w1_s,   w2_s)

# begin energie
E0 = E_s[0]
print(f"  Beginenergie  E₀ = {E0:.4f} J")

# plots

C_EXP  = '#1f77b4'   # blauw  – experiment
C_SIM  = '#d62728'   # rood   – Lagrangiaan theorie
C_TRK  = '#2ca02c'   # groen  – Tracker θ-waarde (validatie)

plt.rcParams.update({
    'font.size'      : 10,
    'axes.titlesize' : 11,
    'figure.dpi'     : 120,
    'lines.linewidth': 1.3,
})

fig = plt.figure(figsize=(16, 20))
gs  = gridspec.GridSpec(4, 2, hspace=0.50, wspace=0.35,
                        left=0.07, right=0.96, top=0.94, bottom=0.04)

# ── hulpfunctie ────────────────────────────────────────────────────
def decor(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.85)

# ─── Plot 1 : θ₁ vs t ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, np.degrees(th_1), color=C_EXP, lw=1.2, label='Experiment (positie)')
ax1.plot(t, np.degrees(th1_s),   color=C_SIM, lw=1.2, ls='--', label='Lagrangiaan')
decor(ax1, 't  (s)', 'θ₁  (°)', 'Hoek lat 1 (scharnier B) vs tijd')

# ─── Plot 2 : θ₂ vs t ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, np.degrees(th_2), color=C_EXP, lw=1.2, label='Experiment')
ax2.plot(t, np.degrees(th2_s),   color=C_SIM, lw=1.2, ls='--', label='Lagrangiaan')
decor(ax2, 't  (s)', 'θ₂  (°)', 'Hoek lat 2 (uiteinde A) vs tijd')

# ─── Plot 3 : Baan massa B ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(xB,   yB,   color=C_EXP, lw=1.0, alpha=0.85, label='Experiment')
ax3.plot(xB_s, yB_s, color=C_SIM, lw=1.0, alpha=0.85, ls='--', label='Lagrangiaan')
ax3.plot(0, 0, 'k^', ms=9, zorder=6, label=f'Draaipunt')

# Begin- en eindpunt markeren
ax3.plot(xB[0],  yB[0],  'bo', ms=6, zorder=5)
ax3.plot(xB[-1], yB[-1], 'bs', ms=6, zorder=5)
ax3.set_aspect('equal')
decor(ax3, 'x  (m)', 'y  (m)', 'Baan massa B (scharnier)')

# ─── Plot 4 : Baan massa A ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(xA,   yA,   color=C_EXP, lw=0.8, alpha=0.85, label='Experiment')
ax4.plot(xA_s, yA_s, color=C_SIM, lw=0.8, alpha=0.85, ls='--', label='Lagrangiaan')
ax4.plot(0, 0, 'k^', ms=9, zorder=6, label='Draaipunt')
ax4.plot(xA[0],  yA[0],  'bo', ms=6, zorder=5, label='t = 0')
ax4.plot(xA[-1], yA[-1], 'bs', ms=6, zorder=5, label='t = einde')
ax4.set_aspect('equal')
decor(ax4, 'x  (m)', 'y  (m)', 'Baan massa A (uiteinde)')

# ─── Plot 5 : Energie experiment ──────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(t, T_e, color='#ff7f0e', lw=1.2, label='Kinetisch  T')
ax5.plot(t, V_e, color='#2ca02c', lw=1.2, label='Potentieel  V')
ax5.plot(t, E_e, color='#1f77b4', lw=2.2, label='Totaal  E = T + V')
ax5.axhline(E0, color='grey', lw=0.8, ls=':', label=f'E₀ = {E0:.3f} J  (theorie)')
decor(ax5, 't  (s)', 'Energie  (J)', 'Energiebalans – Experiment')

# ─── Plot 6 : Energie theorie ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(t, T_s, color='#ff7f0e', lw=1.2, label='Kinetisch  T')
ax6.plot(t, V_s, color='#2ca02c', lw=1.2, label='Potentieel  V')
ax6.plot(t, E_s, color='#1f77b4', lw=2.2, label='Totaal  E = T + V')
ax6.axhline(E0, color='grey', lw=0.8, ls=':', label=f'E₀ = {E0:.3f} J')
decor(ax6, 't  (s)', 'Energie  (J)', 'Energiebalans – Lagrangiaan (RK45)')

# ─── Plot 7 : Faseruimte experiment ───────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
sc7 = ax7.scatter(np.degrees(th_1), np.degrees(th_2),
                  c=t, cmap='plasma', s=2, linewidths=0)
plt.colorbar(sc7, ax=ax7, label='t  (s)', pad=0.02)
decor(ax7, 'θ₁  (°)', 'θ₂  (°)', 'Faseruimte – Experiment')

# ─── Plot 8 : Faseruimte theorie ──────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
sc8 = ax8.scatter(np.degrees(th1_s), np.degrees(th2_s),
                  c=t, cmap='plasma', s=2, linewidths=0)
plt.colorbar(sc8, ax=ax8, label='t  (s)', pad=0.02)
decor(ax8, 'θ₁  (°)', 'θ₂  (°)', 'Faseruimte – Lagrangiaan')

# ─── Titel & opslaan ──────────────────────────────────────────────
fig.suptitle(
    f'Dubbele slinger  ·  Lagrangiaan (puntmassa\'s) vs Experiment\n'
    f'L₁ = {L1*100:.1f} cm, L₂ = {L2*100:.1f} cm  |  '
    f'm₁ = {M1*1000:.0f} g, m₂ = {M2*1000:.0f} g  |  g = {G} m/s²\n'
    f'θ₁(0) = {np.degrees(th1_0):.1f}°,  θ₂(0) = {np.degrees(th2_0):.1f}°  |  '
    f'ω₁(0) = {np.degrees(w1_0):.1f} °/s,  ω₂(0) = {np.degrees(w2_0):.1f} °/s',
    fontsize=11, y=0.975, linespacing=1.6,
)

plt.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
#plt.savefig(OUT_PDF, bbox_inches='tight')
print(f"\n  ✓  PNG opgeslagen → {OUT_PNG}")
#print(f"  ✓  PDF opgeslagen → {OUT_PDF}")

