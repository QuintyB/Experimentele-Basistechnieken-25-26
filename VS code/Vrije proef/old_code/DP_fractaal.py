"""
Chaoskaart van een dubbele slinger
===================================
Berekent de eindige-tijd Lyapunov exponent (FTLE) voor een rooster van
begincondities (θ₁, θ₂) met beginsnelheid nul.

Kleurschaal: zwart = niet-chaotisch | geel → oranje → donkerrood = meer chaos

Vereisten:  pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # verander naar 'TkAgg' of 'Qt5Agg' voor popup
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import gaussian_filter
import time

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  ← pas hier aan
# ─────────────────────────────────────────────────────────────────────────────
L1   = 0.165   # lengte arm 1  [m]
L2   = 0.165   # lengte arm 2  [m]
m1   = 0.069   # massa 1       [kg]
m2   = 0.020   # massa 2       [kg]
g    = 9.81    # zwaartekracht [m/s²]

N          = 1000    # roosterresolutie (300 → ~90 000 punten, ~90 s)
dt         = 0.05    # tijdstap RK4 [s]
T_total    = 30.0    # totale integratietijd [s]
renorm_n   = 5       # renormaliseer elke N stappen
eps        = 1e-9    # grootte initiële perturbatie
smooth_sig = 0       # Gauss-smoothing (0 = uit)
threshold  = 0.35    # FTLE-drempel onder dewelke → zwart (niet-chaotisch)

output_file = "chaos_map_output.png"
dpi         = 220
# ─────────────────────────────────────────────────────────────────────────────



# ── Rooster van begincondities ────────────────────────────────────────────────
thetas      = np.linspace(-np.pi, np.pi, N, endpoint=False)
TH1, TH2    = np.meshgrid(thetas, thetas)
th1_flat    = TH1.ravel()
th2_flat    = TH2.ravel()
Npts        = th1_flat.size

state       = np.zeros((Npts, 4), dtype=np.float64)
state[:, 0] = th1_flat   # θ₁
state[:, 2] = th2_flat   # θ₂
# ω₁ = ω₂ = 0  (beginsnelheid nul)

state_p       = state.copy()
state_p[:, 0] += eps     # kleine perturbatie in θ₁


# ── Bewegingsvergelijkingen (standaardvorm, zie o.a. Wikipedia) ──────────────
def deriv(s):
    th1, w1, th2, w2 = s[:,0], s[:,1], s[:,2], s[:,3]
    d = th1 - th2
    D = 2*m1 + m2 - m2*np.cos(2*d)          # gemeenschappelijke noemer

    dw1 = ( -g*(2*m1 + m2)*np.sin(th1)
             - m2*g*np.sin(th1 - 2*th2)
             - 2*np.sin(d)*m2*(w2**2*L2 + w1**2*L1*np.cos(d))
           ) / (L1 * D)

    dw2 = (  2*np.sin(d)*(  w1**2*L1*(m1 + m2)
                           + g*(m1 + m2)*np.cos(th1)
                           + w2**2*L2*m2*np.cos(d))
           ) / (L2 * D)

    return np.column_stack([w1, dw1, w2, dw2])


# ── RK4 tijdstap ──────────────────────────────────────────────────────────────
def rk4_step(s, h):
    k1 = deriv(s)
    k2 = deriv(s + 0.5*h*k1)
    k3 = deriv(s + 0.5*h*k2)
    k4 = deriv(s +     h*k3)
    return s + h*(k1 + 2*k2 + 2*k3 + k4)/6.0


# ── Lyapunov-integratie ───────────────────────────────────────────────────────
n_steps = int(T_total / dt)
lyap    = np.zeros(Npts, dtype=np.float64)

print(f"Starten: {Npts:,} punten,  {n_steps} stappen,  dt={dt} s")
t0 = time.time()

for i in range(n_steps):
    state   = rk4_step(state,   dt)
    state_p = rk4_step(state_p, dt)

    if (i + 1) % renorm_n == 0:
        diff = state_p - state
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-300)
        lyap += np.log(dist / eps)
        state_p = state + diff * (eps / dist[:, None])   # renormaliseer

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (n_steps - i - 1)
        print(f"  {100*(i+1)/n_steps:.0f}%  stap {i+1}/{n_steps}"
              f"  verstreken={elapsed:.0f}s  ETA={eta:.0f}s")

lyap_map = (lyap / T_total).reshape(N, N)
print(f"\nKlaar in {time.time()-t0:.1f} s")
print(f"FTLE bereik: {lyap_map.min():.3f} … {lyap_map.max():.3f} s⁻¹")


# ── Optionele smoothing ───────────────────────────────────────────────────────
if smooth_sig > 0:
    lyap_plot = gaussian_filter(lyap_map, sigma=smooth_sig)
else:
    lyap_plot = lyap_map.copy()


# ── Kleurkaart: zwart → geel → oranje → donkerrood (omgekeerd) ───────────────
#   Kleine FTLE (weinig chaos) → geel/licht
#   Grote FTLE (veel chaos)    → donkerrood
cmap_nodes = [
    (0.00, (0.00, 0.00, 0.00)),   # zwart  (onder drempel, niet-chaotisch)
    (0.02, (0.15, 0.00, 0.00)),
    (0.10, (0.45, 0.00, 0.00)),   # donker rood  ← grote chaos
    (0.28, (0.72, 0.05, 0.00)),
    (0.50, (0.90, 0.22, 0.00)),
    (0.70, (0.98, 0.50, 0.00)),   # oranje
    (0.87, (1.00, 0.78, 0.00)),
    (1.00, (1.00, 1.00, 0.55)),   # lichtgeel ← kleine (net-chaotische) FTLE
]
cdict = {'red': [], 'green': [], 'blue': []}
for pos, (r, g_c, b) in cmap_nodes:
    cdict['red'  ].append((pos, r,   r))
    cdict['green'].append((pos, g_c, g_c))
    cdict['blue' ].append((pos, b,   b))
fire_r = LinearSegmentedColormap('fire_r', cdict, N=512)
fire_r.set_bad('black')    # niet-chaotisch (NaN) → zwart

# Maskeer niet-chaotisch gebied
display = lyap_plot.copy()
display[lyap_plot < threshold] = np.nan

# Logaritmische normalisatie: beste spreiding over het dynamisch bereik
norm = LogNorm(vmin=threshold, vmax=lyap_map.max())


# ── Plot ──────────────────────────────────────────────────────────────────────
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

ax.set_xlim(-180, 180);  ax.set_ylim(-180, 180)
ax.set_xticks(deg_ticks[::2]);  ax.set_yticks(deg_ticks[::2])
ax.tick_params(colors='white', labelsize=9.5, direction='in',
               length=5, width=0.8, top=True, right=True)
for spine in ax.spines.values():
    spine.set_edgecolor('white');  spine.set_linewidth(0.8)

ax.set_xlabel(r'$\alpha_0$  (graden)', color='white', fontsize=14, labelpad=4)
ax.set_ylabel(r'$\beta_0$  (graden)',  color='white', fontsize=14, labelpad=4)

cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015, aspect=30)
cbar.set_label('Lyapunov exponent  λ [s⁻¹]', color='white', fontsize=9)
cbar.ax.tick_params(colors='white', labelsize=8)
cbar.outline.set_edgecolor('white');  cbar.outline.set_linewidth(0.8)

ax.text(0.98, 0.98,
        f'$L_1 = L_2 = {L1*100:.1f}$ cm\n'
        f'$m_1 = {m1*1000:.0f}$ g,  $m_2 = {m2*1000:.0f}$ g\n'
        f'$T = {T_total:.0f}$ s,  $\\Delta t = {dt}$ s',
        transform=ax.transAxes, color='white', fontsize=8,
        va='top', ha='right',
        bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

plt.tight_layout(pad=0.5)
plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
            facecolor='black', edgecolor='black')
plt.close()
print(f"\nFiguur opgeslagen als '{output_file}'")