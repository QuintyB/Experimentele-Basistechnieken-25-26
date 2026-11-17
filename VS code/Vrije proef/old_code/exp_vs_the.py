"""
Chaoskaart van een dubbele slinger + experimentele beginpunten
================================================================
1. Berekent de FTLE (eindige-tijd Lyapunov exponent) voor een rooster van
   begincondities (θ₁, θ₂) – zwart = niet-chaotisch, geel→rood = chaos.
2. Leest de experimentele Tracker-data in, bepaalt de beginhoeken (θ₁, θ₂)
   en zet die als gekleurde symbolen op de kaart:
       ● cyaan  = experimenteel chaotisch   (FTLE ≥ drempel)
       ● groen  = experimenteel niet-chaotisch (FTLE < drempel)

Vereisten:  pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import time
import os

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  ← pas hier aan
# ═══════════════════════════════════════════════════════════════════════════════

# ── Slinger-fysica ─────────────────────────────────────────────────────────────
L1   = 0.165   # lengte arm 1  [m]
L2   = 0.165   # lengte arm 2  [m]
m1   = 0.069   # massa 1       [kg]
m2   = 0.020   # massa 2       [kg]
g    = 9.81    # zwaartekracht [m/s²]

# ── Chaos-kaart ────────────────────────────────────────────────────────────────
N          = 300     # roosterresolutie (300 → ~90 000 punten)
dt         = 0.05    # tijdstap RK4 [s]
T_total    = 30.0    # totale integratietijd [s]
renorm_n   = 5       # renormaliseer elke N stappen
eps        = 1e-9    # grootte initiële perturbatie
smooth_sig = 0.8     # Gauss-smoothing (0 = uit)
threshold  = 0.35    # FTLE-drempel: onder deze waarde → zwart (niet-chaotisch)

# ── Experimentele data ─────────────────────────────────────────────────────────
BASE       = '/Users/swinn/Documents/2de bachelor/experimentel basistechnieken/VS code/vrij proef'
N_PROEVEN  = 12      # aantal proeven (1 t/m N_PROEVEN)
N_FRAMES   = 600     # aantal frames per proef dat wordt gebruikt
FPS_FACTOR = 2       # tijdas gedeeld door deze factor (slowmotion-correctie)

# ── Output ─────────────────────────────────────────────────────────────────────
output_file = "chaos_map_experiment.png"
dpi         = 220

# ═══════════════════════════════════════════════════════════════════════════════
# DEEL 1 – CHAOS-KAART (FTLE)
# ═══════════════════════════════════════════════════════════════════════════════

thetas   = np.linspace(-np.pi, np.pi, N, endpoint=False)
TH1, TH2 = np.meshgrid(thetas, thetas)
th1_flat = TH1.ravel()
th2_flat = TH2.ravel()
Npts     = th1_flat.size

state       = np.zeros((Npts, 4), dtype=np.float64)
state[:, 0] = th1_flat
state[:, 2] = th2_flat

state_p       = state.copy()
state_p[:, 0] += eps


def deriv(s):
    th1, w1, th2, w2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    d = th1 - th2
    D = 2 * m1 + m2 - m2 * np.cos(2 * d)

    dw1 = (
        -g * (2 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2 * th2)
        - 2 * np.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * np.cos(d))
    ) / (L1 * D)

    dw2 = (
        2 * np.sin(d) * (
            w1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + w2**2 * L2 * m2 * np.cos(d)
        )
    ) / (L2 * D)

    return np.column_stack([w1, dw1, w2, dw2])


def rk4_step(s, h):
    k1 = deriv(s)
    k2 = deriv(s + 0.5 * h * k1)
    k3 = deriv(s + 0.5 * h * k2)
    k4 = deriv(s +       h * k3)
    return s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


n_steps = int(T_total / dt)
lyap    = np.zeros(Npts, dtype=np.float64)

print(f"Chaos-kaart: {Npts:,} punten,  {n_steps} stappen,  dt={dt} s")
t0 = time.time()

for i in range(n_steps):
    state   = rk4_step(state,   dt)
    state_p = rk4_step(state_p, dt)

    if (i + 1) % renorm_n == 0:
        diff = state_p - state
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-300)
        lyap += np.log(dist / eps)
        state_p = state + diff * (eps / dist[:, None])

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (n_steps - i - 1)
        print(f"  {100*(i+1)/n_steps:.0f}%  stap {i+1}/{n_steps}"
              f"  verstreken={elapsed:.0f}s  ETA={eta:.0f}s")

lyap_map = (lyap / T_total).reshape(N, N)
print(f"\nKlaar in {time.time() - t0:.1f} s")
print(f"FTLE bereik: {lyap_map.min():.3f} … {lyap_map.max():.3f} s⁻¹")

if smooth_sig > 0:
    lyap_plot = gaussian_filter(lyap_map, sigma=smooth_sig)
else:
    lyap_plot = lyap_map.copy()
 
# ═══════════════════════════════════════════════════════════════════════════════
# DEEL 2 – EXPERIMENTELE BEGINHOEKEN INLEZEN
# ═══════════════════════════════════════════════════════════════════════════════

def read_tracker(path, ncols=3):
    """Leest Tracker-exportbestand (tab-gescheiden, komma als decimaalteken)."""
    rows = []
    with open(path, encoding='utf-8') as fh:
        for idx, line in enumerate(fh):
            if idx < 2:
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


# Sla de experimentele beginhoeken op (in graden)
exp_th1_deg = []   # beginhoek arm 1 per proef
exp_th2_deg = []   # beginhoek arm 2 per proef
exp_labels  = []   # proefnummer als label

print("\nExperimentele data inlezen …")
for i in range(1, N_PROEVEN + 1):
    file_a = f'{BASE}/data/data proef twee/proef_2_{i}_A.txt'
    file_b = f'{BASE}/data/data proef twee/proef_2_{i}_B.txt'

    if not (os.path.exists(file_a) and os.path.exists(file_b)):
        print(f"  Proef {i}: bestanden niet gevonden – overgeslagen.")
        continue

    try:
        raw_b = read_tracker(file_b, 3)
        raw_a = read_tracker(file_a, 3)
    except Exception as e:
        print(f"  Proef {i}: leesfout ({e}) – overgeslagen.")
        continue

    n   = min(len(raw_b), len(raw_a), N_FRAMES)
    t   = raw_b[:n, 0] / FPS_FACTOR

    xB, yB = raw_b[:n, 1], raw_b[:n, 2]
    xA, yA = raw_a[:n, 1], raw_a[:n, 2]

    # Continue hoeken (unwrap) → beginwaarde
    th1_cont = np.unwrap(np.arctan2(xB, -yB))
    th2_cont = np.unwrap(np.arctan2(xA - xB, yB - yA))

    th1_0 = wrap_angle(th1_cont[0])
    th2_0 = wrap_angle(th2_cont[0])

    exp_th1_deg.append(np.degrees(th1_0))
    exp_th2_deg.append(np.degrees(th2_0))
    exp_labels.append(str(i))
    print(f"  Proef {i}: θ₁(0) = {np.degrees(th1_0):+6.1f}°,  θ₂(0) = {np.degrees(th2_0):+6.1f}°")

exp_th1_deg = np.array(exp_th1_deg)
exp_th2_deg = np.array(exp_th2_deg)

# Zoek voor elk experimenteel punt de FTLE-waarde op in de kaart
def ftle_at(th1_deg, th2_deg):
    """Geeft de (geïnterpoleerde) FTLE-waarde op (th1_deg, th2_deg)."""
    j = int(np.round((th1_deg + 180) / 360 * N)) % N   # kolom (θ₁-as = x)
    i = int(np.round((th2_deg + 180) / 360 * N)) % N   # rij   (θ₂-as = y)
    return lyap_plot[i, j]

if len(exp_th1_deg) > 0:
    ftle_vals = np.array([ftle_at(a, b) for a, b in zip(exp_th1_deg, exp_th2_deg)])
    is_chaotic = ftle_vals >= threshold
    print(f"\nExperimentele punten gevonden: {len(exp_th1_deg)}")
    print(f"  Chaotisch (FTLE ≥ {threshold}):     {is_chaotic.sum()}")
    print(f"  Niet-chaotisch (FTLE < {threshold}): {(~is_chaotic).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# DEEL 3 – FIGUUR
# ═══════════════════════════════════════════════════════════════════════════════

# Kleurkaart: zwart → donkerrood → oranje → lichtgeel
cmap_nodes = [
    (0.00, (0.00, 0.00, 0.00)),
    (0.02, (0.15, 0.00, 0.00)),
    (0.10, (0.45, 0.00, 0.00)),
    (0.28, (0.72, 0.05, 0.00)),
    (0.50, (0.90, 0.22, 0.00)),
    (0.70, (0.98, 0.50, 0.00)),
    (0.87, (1.00, 0.78, 0.00)),
    (1.00, (1.00, 1.00, 0.55)),
]
cdict = {'red': [], 'green': [], 'blue': []}
for pos, (r, gc, b) in cmap_nodes:
    cdict['red'  ].append((pos, r,  r))
    cdict['green'].append((pos, gc, gc))
    cdict['blue' ].append((pos, b,  b))
fire_r = LinearSegmentedColormap('fire_r', cdict, N=512)
fire_r.set_bad('black')

display = lyap_plot.copy()
display[lyap_plot < threshold] = np.nan
norm = LogNorm(vmin=threshold, vmax=lyap_map.max())

# ── Teken figuur ───────────────────────────────────────────────────────────────
deg_ticks = np.arange(-180, 181, 30)

fig, ax = plt.subplots(figsize=(8.5, 8.0), facecolor='black')
ax.set_facecolor('black')

im = ax.imshow(display,
               origin='lower',
               extent=[-180, 180, -180, 180],
               cmap=fire_r,
               norm=norm,
               interpolation='bilinear',
               aspect='equal')

# ── Experimentele punten ───────────────────────────────────────────────────────
COLOR_CHAOS    = '#00FFEE'   # cyaan  – experimenteel chaotisch
COLOR_NOCHAOS  = '#39FF14'   # neongeel-groen – experimenteel niet-chaotisch
MARKER_SIZE    = 120         # markergrootte (scatter s=)
EDGE_COLOR     = 'white'
EDGE_WIDTH     = 0.8

if len(exp_th1_deg) > 0:
    # Niet-chaotisch (groen) eerst, daarna chaotisch (cyaan) erop
    mask_nc = ~is_chaotic
    mask_c  =  is_chaotic

    if mask_nc.any():
        ax.scatter(exp_th1_deg[mask_nc], exp_th2_deg[mask_nc],
                   s=MARKER_SIZE, c=COLOR_NOCHAOS, marker='o',
                   edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
                   zorder=5, label='Experiment – niet chaotisch')

    if mask_c.any():
        ax.scatter(exp_th1_deg[mask_c], exp_th2_deg[mask_c],
                   s=MARKER_SIZE, c=COLOR_CHAOS, marker='o',
                   edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
                   zorder=5, label='Experiment – chaotisch')

    # Proefnummers als kleine annotatie naast elk punt
    for x, y, lbl in zip(exp_th1_deg, exp_th2_deg, exp_labels):
        ax.annotate(lbl, (x, y),
                    xytext=(5, 5), textcoords='offset points',
                    color='white', fontsize=7, fontweight='bold',
                    zorder=6)

# ── Assen & opmaak ─────────────────────────────────────────────────────────────
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks(deg_ticks[::2])
ax.set_yticks(deg_ticks[::2])
ax.tick_params(colors='white', labelsize=9.5, direction='in',
               length=5, width=0.8, top=True, right=True)
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(0.8)

ax.set_xlabel(r'$\theta_1(0)$  (graden)', color='white', fontsize=14, labelpad=4)
ax.set_ylabel(r'$\theta_2(0)$  (graden)', color='white', fontsize=14, labelpad=4)

# Kleurenbalk FTLE
cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015, aspect=30)
cbar.set_label('Lyapunov exponent  λ  [s⁻¹]', color='white', fontsize=9)
cbar.ax.tick_params(colors='white', labelsize=8)
cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(0.8)

# Parameterinfo rechts boven
ax.text(0.98, 0.98,
        f'$L_1 = L_2 = {L1*100:.1f}$ cm\n'
        f'$m_1 = {m1*1000:.0f}$ g,  $m_2 = {m2*1000:.0f}$ g\n'
        f'$T = {T_total:.0f}$ s,  $\\Delta t = {dt}$ s',
        transform=ax.transAxes, color='white', fontsize=8,
        va='top', ha='right',
        bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

# Legenda experimentele punten
if len(exp_th1_deg) > 0:
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_CHAOS, markeredgecolor=EDGE_COLOR,
               markeredgewidth=EDGE_WIDTH, markersize=9,
               label='Experiment – chaotisch'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_NOCHAOS, markeredgecolor=EDGE_COLOR,
               markeredgewidth=EDGE_WIDTH, markersize=9,
               label='Experiment – niet-chaotisch'),
    ]
    leg = ax.legend(handles=legend_elements,
                    loc='lower right',
                    fontsize=8.5,
                    framealpha=0.55,
                    facecolor='black',
                    edgecolor='white',
                    labelcolor='white')

plt.tight_layout(pad=0.5)
plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
            facecolor='black', edgecolor='black')
plt.close()
print(f"\nFiguur opgeslagen als '{output_file}'")