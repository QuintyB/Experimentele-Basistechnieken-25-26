"""
╔══════════════════════════════════════════════════════════════════════╗
║           CHAOS-KAART VAN EEN DUBBELE SLINGER                       ║
║           op basis van de Lagrangiaan-bewegingsvergelijkingen        ║
╚══════════════════════════════════════════════════════════════════════╝

WAT DOET DIT SCRIPT?
────────────────────
1. We maken een groot rooster van beginhoeken (θ₁, θ₂) — elk punt
   stelt een mogelijke beginpositie van de dubbele slinger voor.

2. Voor elk beginpunt simuleren we tegelijk twee bijna-identieke slingers:
   één origineel en één met een piepkleine verstoring (eps = 1e-9 rad).

3. We meten hoe snel die twee banen uit elkaar groeien over de tijd.
   Die groeisnelheid heet de "Lyapunov exponent" (FTLE):
     - Groot FTLE  →  banen groeien snel uit elkaar  →  CHAOTISCH  (geel/rood)
     - Klein FTLE  →  banen blijven dicht bij elkaar  →  STABIEL    (zwart)

4. We lezen de experimentele beginhoeken in vanuit de Tracker-data
   en tekenen die als gekleurde punten op de kaart.
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# numpy      : wiskundige bewerkingen op arrays (vectoren, matrices)
# matplotlib : tekenen van figuren en grafieken
#   .colors  : speciale kleurkaarten en normalisaties
#   .lines   : voor de legenda-symbolen
# os         : controleren of bestanden bestaan op de harde schijf
# ══════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib
matplotlib.use('Agg')          # geen pop-upvenster, sla direct op als bestand
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import os


# ══════════════════════════════════════════════════════════════════════
# PARAMETERS  ← hier pas je de instellingen aan
# ══════════════════════════════════════════════════════════════════════

# Fysische eigenschappen van de slinger
M1 = 0.069   # massa van het eerste gewicht [kg]
M2 = 0.020   # massa van het tweede gewicht [kg]
G  = 9.81    # zwaartekrachtversnelling [m/s²]
L1 = 0.165   # lengte van de eerste arm [m]
L2 = 0.165   # lengte van de tweede arm [m]

# Instellingen voor de chaos-kaart
N         = 1080    # aantal roosterpunten per as → totaal N² beginpunten
                   # 200 → 40 000 punten (~2 min rekentijd)
                   # 300 → 90 000 punten (~5 min rekentijd, meer detail)
dt        = 0.05   # grootte van één tijdstap in de simulatie [s]
                   # kleiner = nauwkeuriger maar trager
T_total   = 30.0   # hoe lang we elke slinger simuleren [s]
                   # langer = betrouwbaardere FTLE-waarden
eps       = 1e-9   # begingrootte van de verstoring [rad]
                   # heel klein zodat we echt de lokale chaos meten
threshold = 0.25   # FTLE-grenswaarde [s⁻¹]
                   # punten ONDER deze waarde worden zwart (= niet-chaotisch)
                   # punten BOVEN deze waarde krijgen een kleur (= chaotisch)

# Bestandspaden voor de experimentele data
BASE        = '/Users/swinn/Documents/2de bachelor/experimentel basistechnieken/VS code/vrij proef'
N_PROEVEN   = 32           # hoeveel proeven er zijn (proef_2_1 t/m proef_2_{'N_PROEVEN'})
output_file = 'eigen_chaos_kaart.png'   # naam van de uitvoerfiguur
dpi         = 220          # resolutie van de figuur (pixels per inch)


# ══════════════════════════════════════════════════════════════════════
# STAP 1 – BEWEGINGSVERGELIJKINGEN (Lagrangiaan)
# ══════════════════════════════════════════════════════════════════════
#
# De dubbele slinger heeft 2 vrijheidsgraden: θ₁ (hoek arm 1) en θ₂ (hoek arm 2).
# De toestand van het systeem wordt volledig beschreven door 4 variabelen:
#   [θ₁,  ω₁,  θ₂,  ω₂]
#    hoek snelh hoek snelh
#    arm1 arm1  arm2 arm2
#
# De bewegingsvergelijkingen komen uit de Lagrangiaan L = T - V
# (kinetische energie min potentiële energie).
# Ze hebben de vorm van een matrixvergelijking:
#
#   [M11  M12] [α₁]   [f1]
#   [M12  M22] [α₂] = [f2]
#
# Waarbij α₁ en α₂ de hoekversnellingen zijn die we willen berekenen.
# We lossen dit analytisch op voor een 2×2 matrix.
#
# BELANGRIJK: deze functie werkt niet op 1 punt maar op een ARRAY van
# N² punten tegelijk. Elke rij van s stelt één slingertoestand voor.
# Dit is exact dezelfde wiskunde als in eigen_code.py, maar dan
# voor 40 000 punten in één keer → veel sneller dan een for-lus.

def deriv(s):
    """
    Berekent de tijdsafgeleiden van de toestand voor alle roosterpunten tegelijk.

    Invoer:
      s  →  array met vorm (N², 4)
            elke rij = één toestand: [θ₁, ω₁, θ₂, ω₂]

    Uitvoer:
      array met vorm (N², 4)
      elke rij = de afgeleiden: [ω₁, α₁, ω₂, α₂]
      (dθ₁/dt = ω₁,  dω₁/dt = α₁,  dθ₂/dt = ω₂,  dω₂/dt = α₂)
    """

    # Haal elke kolom apart op als array van N² waarden
    th1, w1, th2, w2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

    # Hoeksverschil tussen de twee armen — komt voor in de massamatrix
    delta = th1 - th2
    cd = np.cos(delta)   # cosinus van het verschil, array van N² waarden
    sd = np.sin(delta)   # sinus  van het verschil, array van N² waarden

    # ── Massamatrix ──────────────────────────────────────────────────
    # M11 en M22 zijn scalars (hetzelfde voor alle punten)
    # M12 is een array omdat cd per punt verschilt
    M11 = (M1 + M2) * L1**2        # scalar
    M12 = M2 * L1 * L2 * cd        # array (N² waarden)
    M22 = M2 * L2**2                # scalar

    # ── Rechterlid (gegeneraliseerde krachten) ────────────────────────
    # f1 en f2 zijn de "aandrijvende krachten" in de vergelijking M·α = f
    # Ze bevatten zwaartekracht (sinus-termen) en middelpuntzoekende krachten
    # (de ω²-termen die komen door de koppeling tussen de twee armen)
    f1 = (-M2 * L1 * L2 * sd * w2**2        # koppelingsterm arm2 → arm1
          - (M1 + M2) * G * L1 * np.sin(th1))  # zwaartekracht op arm1
    f2 = (+M2 * L1 * L2 * sd * w1**2        # koppelingsterm arm1 → arm2
          - M2 * G * L2 * np.sin(th2))          # zwaartekracht op arm2

    # ── Analytische oplossing van M·α = f voor 2×2 matrix ────────────
    # Voor een 2×2 matrix [a b; c d] geldt: inverse = 1/det * [d -b; -c a]
    # met det = a*d - b*c
    det  = M11 * M22 - M12**2       # determinant, array van N² waarden
    alp1 = (M22 * f1 - M12 * f2) / det   # hoekversnelling arm 1
    alp2 = (M11 * f2 - M12 * f1) / det   # hoekversnelling arm 2

    # Geef de afgeleiden terug als array met 4 kolommen:
    # kolom 0: dθ₁/dt = ω₁  (hoeksnelheid is de afgeleide van de hoek)
    # kolom 1: dω₁/dt = α₁  (hoekversnelling is de afgeleide van de snelheid)
    # kolom 2: dθ₂/dt = ω₂
    # kolom 3: dω₂/dt = α₂
    return np.column_stack([w1, alp1, w2, alp2])


def rk4_stap(s, h):
    """
    Berekent één tijdstap vooruit met de Runge-Kutta 4 methode (RK4).

    RK4 is een numerieke integratiemethode: gegeven de huidige toestand s
    op tijd t, bereken de toestand op tijd t+h.
    Het gebruikt 4 tussenliggende schattingen (k1 t/m k4) om een
    nauwkeurigere stap te maken dan simpele Euler-integratie.

      k1 = afgeleide aan het begin van de stap
      k2 = afgeleide halverwege, met k1 als schatting
      k3 = afgeleide halverwege, met k2 als schatting (verfijnd)
      k4 = afgeleide aan het einde van de stap

    De gewogen combinatie (1*k1 + 2*k2 + 2*k3 + 1*k4)/6 geeft
    een fout van orde h⁴ per stap (= zeer nauwkeurig).

    Invoer:  s = huidige toestand (N², 4),  h = tijdstap [s]
    Uitvoer: nieuwe toestand na tijd h
    """
    k1 = deriv(s)
    k2 = deriv(s + 0.5 * h * k1)
    k3 = deriv(s + 0.5 * h * k2)
    k4 = deriv(s +       h * k3)
    return s + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# ══════════════════════════════════════════════════════════════════════
# STAP 2 – CHAOS-KAART BEREKENEN
# ══════════════════════════════════════════════════════════════════════

print(f"Chaos-kaart: {N*N:,} punten,  {int(T_total/dt)} stappen")

# ── Rooster van beginhoeken ───────────────────────────────────────────
# linspace(-π, π, N) maakt N gelijkmatig verdeelde hoeken van -180° tot +180°
# meshgrid zet twee 1D-arrays om naar een 2D-rooster:
#   TH1[i,j] = beginhoek θ₁ voor roosterpunt (i,j)
#   TH2[i,j] = beginhoek θ₂ voor roosterpunt (i,j)
thetas = np.linspace(-np.pi, np.pi, N, endpoint=False)
TH1, TH2 = np.meshgrid(thetas, thetas)

# ── Begintoestand voor alle N² punten ────────────────────────────────
# We starten overal met ω₁ = ω₂ = 0 (de slinger staat stil)
# .ravel() maakt van een 2D-rooster (N,N) een 1D-array van lengte N²
state = np.zeros((N * N, 4))    # vorm: (N², 4)
state[:, 0] = TH1.ravel()       # kolom 0 = θ₁ voor elk beginpunt
state[:, 2] = TH2.ravel()       # kolom 2 = θ₂ voor elk beginpunt
# kolommen 1 en 3 (ω₁ en ω₂) blijven 0

# ── Verstoorde begintoestand ──────────────────────────────────────────
# Voor elk punt maken we een tweede, bijna-identieke slinger
# waarvan θ₁ een heel klein beetje groter is (eps = 1e-9 rad)
# Als de twee banen snel uit elkaar gaan → chaotisch
state_p = state.copy()          # kopie van de begintoestand
state_p[:, 0] += eps            # voeg een kleine verstoring toe aan θ₁

# ── Lyapunov-accumulatie ──────────────────────────────────────────────
# lyap accumuleert log(groei) over de tijd
# aan het einde delen we door T_total om de FTLE te krijgen
lyap    = np.zeros(N * N)
n_steps = int(T_total / dt)     # totaal aantal tijdstappen

for i in range(n_steps):

    # Zet beide toestandsarrays één tijdstap vooruit
    state   = rk4_stap(state,   dt)
    state_p = rk4_stap(state_p, dt)

    # Elke 5 stappen: meet en renormaliseer
    # (vaker renormaliseren = stabieler, minder vaak = sneller)
    if (i + 1) % 5 == 0:

        # Verschilvector tussen origineel en verstoord punt
        diff = state_p - state              # array (N², 4)

        # Afstand in de 4D-toestandsruimte (norm over de 4 variabelen)
        dist = np.linalg.norm(diff, axis=1) # array (N²,)
        dist = np.maximum(dist, 1e-300)     # voorkom log(0) bij numerieke precisie

        # Tel de logaritmische groei op
        # Als de afstand verdubbeld is: log(2*eps/eps) = log(2) ≈ 0.69
        lyap += np.log(dist / eps)

        # Renormaliseer: zet state_p terug zodat de afstand weer eps is
        # richting blijft hetzelfde, alleen de lengte wordt hersteld
        state_p = state + diff * (eps / dist[:, None])

    # Voortgangsindicator: print elke 100 stappen het percentage
    if (i + 1) % 10 == 0:
        print(f"  {100*(i+1)/n_steps:.0f}%")

# ── FTLE berekenen ────────────────────────────────────────────────────
# Deel door T_total om de gemiddelde groeisnelheid per seconde te krijgen
# .reshape(N, N) zet de 1D-array van N² waarden terug naar een 2D-kaart
lyap_map  = (lyap / T_total).reshape(N, N)
lyap_plot = lyap_map.copy()     # kopie die we later voor het figuur gebruiken

print(f"Klaar.  FTLE-bereik: {lyap_map.min():.3f} ... {lyap_map.max():.3f} s-1")


# ══════════════════════════════════════════════════════════════════════
# STAP 3 – EXPERIMENTELE BEGINHOEKEN INLEZEN
# ══════════════════════════════════════════════════════════════════════

def lees_tracker(pad, ncols=3):
    """
    Leest een Tracker-exportbestand in en geeft een numpy-array terug.

    Tracker exporteert data als tekst met:
      - 2 headerregels bovenaan (die slaan we over)
      - tab-gescheiden kolommen
      - komma als decimaalteken (Nederlands formaat → we vervangen , door .)

    Invoer:
      pad   → bestandspad als string
      ncols → hoeveel kolommen we inlezen (standaard 3: t, x, y)

    Uitvoer:
      numpy-array met vorm (aantal_frames, ncols)
    """
    rijen = []
    with open(pad, encoding='utf-8') as f:
        for nr, lijn in enumerate(f):
            if nr < 2:              # sla de 2 headerregels over
                continue
            lijn = lijn.strip()     # verwijder witruimte aan begin en einde
            if not lijn:            # sla lege regels over
                continue
            delen = lijn.replace(',', '.').split('\t')  # komma→punt, splits op tab
            try:
                rijen.append([float(delen[j]) for j in range(ncols)])
            except (ValueError, IndexError):
                pass                # sla regels over die niet gelezen kunnen worden
    return np.array(rijen)


def wrap(theta):
    """
    Vouwt een hoek terug naar het interval [-π, π] (= -180° tot +180°).

    np.unwrap geeft soms hoeken buiten dit bereik (bv. 400° of -250°).
    Deze functie brengt ze terug naar de "standaard" voorstelling.

    Voorbeeld: wrap(3.5π) = -0.5π  (want 3.5π - 2π = 1.5π > π, dus nog eens - 2π = -0.5π)
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


# Lijsten om de experimentele beginhoeken en proefnummers bij te houden
exp_th1_deg = []   # beginhoek arm 1 per proef [graden]
exp_th2_deg = []   # beginhoek arm 2 per proef [graden]
exp_labels  = []   # proefnummer als tekst (voor annotaties op de kaart)
choatic =[]

print("\nExperimentele data inlezen ...")
for i in range(1, N_PROEVEN + 1):
    pad_a = f'Vrije proef/data/proef_2_{i}_A.txt'   # uiteinde (massa A)
    pad_b = f'Vrije proef/data/proef_2_{i}_B.txt'   # scharnier (massa B)

    # Controleer of beide bestanden bestaan, anders sla de proef over
    if not (os.path.exists(pad_a) and os.path.exists(pad_b)):
        continue

    raw_b = lees_tracker(pad_b)   # data voor punt B: kolommen [t, xB, yB]
    raw_a = lees_tracker(pad_a)   # data voor punt A: kolommen [t, xA, yA]

    # Gebruik maximaal 600 frames (= N_FRAMES in de originele code)
    n = min(len(raw_b), len(raw_a), 600)

    xB, yB = raw_b[:n, 1], raw_b[:n, 2]   # x- en y-coördinaten van scharnier B
    xA, yA = raw_a[:n, 1], raw_a[:n, 2]   # x- en y-coördinaten van uiteinde A

    ## controleren of hij een volledig rondje doet
    theta2           = np.arctan2(xA - xB, yB - yA)   # hoek van de onderste arm
    theta2_unwrapped = np.unwrap(theta2)
    range_rotation   = theta2_unwrapped.max() - theta2_unwrapped.min()
    is_chaotic_i     = range_rotation >= 2 * np.pi
    choatic.append(is_chaotic_i)


    # ── Hoekberekening ────────────────────────────────────────────────
    # arctan2(x, -y) geeft de hoek van de arm t.o.v. de negatieve y-as (= neerwaarts)
    # np.unwrap zorgt dat de hoek continu blijft (geen sprongen van -π naar +π)
    # [0] pakt alleen het eerste frame = beginhoek
    th1_0 = wrap(np.unwrap(np.arctan2(xB, -yB))[0])          # beginhoek arm 1
    th2_0 = wrap(np.unwrap(np.arctan2(xA - xB, yB - yA))[0]) # beginhoek arm 2

    # Sla op in graden (voor de plot, die werkt in graden)
    exp_th1_deg.append(np.degrees(th1_0))
    exp_th2_deg.append(np.degrees(th2_0))
    exp_labels.append(str(i))
    print(f"  Proef {i}: th1={np.degrees(th1_0):+.1f} deg, th2={np.degrees(th2_0):+.1f} deg")

# Zet de lijsten om naar numpy-arrays (nodig voor vectoroperaties in de plot)
exp_th1_deg = np.array(exp_th1_deg)
exp_th2_deg = np.array(exp_th2_deg)
is_chaotic = np.array(choatic, dtype=bool)



# ══════════════════════════════════════════════════════════════════════
# STAP 4 – FIGUUR
# ══════════════════════════════════════════════════════════════════════

# ── Kleurkaart bouwen ─────────────────────────────────────────────────
# We bouwen een aangepaste kleurkaart: zwart (niet-chaotisch) →
# donkerrood → oranje → lichtgeel (sterk chaotisch)
#
# cmap_nodes is een lijst van (positie, (r, g, b)) koppels:
#   positie = 0.0 (begin van de schaal) tot 1.0 (einde van de schaal)
#   (r, g, b) = kleur op die positie, elk tussen 0.0 en 1.0
#
# LinearSegmentedColormap interpoleert lineair tussen de knooppunten
cmap_nodes = [
    (0.00, (0.00, 0.00, 0.00)),   # zwart       (laagste FTLE)
    (0.02, (0.15, 0.00, 0.00)),   # zeer donker rood
    (0.10, (0.45, 0.00, 0.00)),   # donker rood
    (0.28, (0.72, 0.05, 0.00)),   # rood
    (0.50, (0.90, 0.22, 0.00)),   # rood-oranje
    (0.70, (0.98, 0.50, 0.00)),   # oranje
    (0.87, (1.00, 0.78, 0.00)),   # geel-oranje
    (1.00, (1.00, 1.00, 0.55)),   # lichtgeel   (hoogste FTLE)
]

# Bouw het cdict-woordenboek dat LinearSegmentedColormap verwacht:
# elk kanaal (rood, groen, blauw) krijgt een lijst van (positie, waarde, waarde)
cdict = {'red': [], 'green': [], 'blue': []}
for pos, (r, gc, b) in cmap_nodes:
    cdict['red'  ].append((pos, r,  r))
    cdict['green'].append((pos, gc, gc))
    cdict['blue' ].append((pos, b,  b))

fire_r = LinearSegmentedColormap('fire_r', cdict, N=512)
fire_r.set_bad('black')   # NaN-waarden (niet-chaotisch) worden zwart ingekleurd

# ── Beelddata voorbereiden ────────────────────────────────────────────
# Maak een kopie van lyap_plot en zet alle waarden ONDER de drempel op NaN.
# NaN = "geen geldige waarde" → matplotlib kleurt die zwart (zie fire_r.set_bad)
display = lyap_plot.copy()
display[lyap_plot < threshold] = np.nan

# LogNorm schaalt de kleurkaart logaritmisch:
# kleine FTLE-verschillen nabij de drempel zijn ook zichtbaar, niet alleen extreme waarden
norm = LogNorm(vmin=threshold, vmax=lyap_map.max())

deg_ticks = np.arange(-180, 181, 30)   # astitels: -180, -150, ..., +180

# ── Figuur aanmaken ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 8.0), facecolor='black')
ax.set_facecolor('black')

# imshow tekent de 2D-array als een afbeelding:
#   origin='lower'          → θ₂ = -180° onderaan, +180° bovenaan
#   extent=[-180,180,-180,180] → de assen lopen van -180° tot +180°
#   cmap=fire_r             → gebruik onze zelfgemaakte kleurkaart
#   norm=norm               → logaritmische kleurschaal
#   interpolation='bilinear' → vloeiende overgang tussen roostercellen
#   aspect='equal'          → gelijke schaal op beide assen
im = ax.imshow(display,
               origin='lower',
               extent=[-180, 180, -180, 180],
               cmap=fire_r,
               norm=norm,
               interpolation='bilinear',
               aspect='equal')

# ── Experimentele punten op de kaart zetten ───────────────────────────
COLOR_CHAOS   = '#00FFEE'   # cyaan  → experimenteel chaotisch
COLOR_NOCHAOS = '#39FF14'   # neongeel-groen → experimenteel niet-chaotisch
MARKER_SIZE   = 120         # markeroppervlak in punten² (scatter s=)
EDGE_COLOR    = 'white'     # witte rand om de markers (beter zichtbaar)
EDGE_WIDTH    = 0.8




mask_nc = ~is_chaotic   # masker voor niet-chaotische punten
mask_c  =  is_chaotic   # masker voor chaotische punten

    # Teken eerst de groene punten (niet-chaotisch), dan de cyaane eroverheen
if mask_nc.any():
        ax.scatter(exp_th1_deg[mask_nc], exp_th2_deg[mask_nc],
                   s=MARKER_SIZE, c=COLOR_NOCHAOS, marker='o',
                   edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=5)

if mask_c.any():
        ax.scatter(exp_th1_deg[mask_c], exp_th2_deg[mask_c],
                   s=MARKER_SIZE, c=COLOR_CHAOS, marker='o',
                   edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=5)

   


# ── Assen en opmaak ───────────────────────────────────────────────────
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks(deg_ticks[::2])    # toon elke 2e tick (elke 60°)
ax.set_yticks(deg_ticks[::2])
ax.tick_params(colors='black', labelsize=15, direction='in',
               length=5, width=0.8, top=True, right=True)
for spine in ax.spines.values():  # witte kader rondom de plot
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

ax.set_xlabel(r'$\theta_1(0)$  [$^\circ$]', color='black', fontsize=25, labelpad=4)
ax.set_ylabel(r'$\theta_2(0)$  [$^\circ$]', color='black', fontsize=25, labelpad=4)

# ── Kleurenbalk (colorbar) ────────────────────────────────────────────
# Toont de koppeling tussen kleur en FTLE-waarde
cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015, aspect=30)
cbar.set_label(r'Lyapunov exponent  $\lambda$  [s$^{-1}$]', color='black', fontsize=25)
cbar.ax.tick_params(colors='black', labelsize=8)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(0.8)

# ── Parameterbox rechts boven ─────────────────────────────────────────
# Kleine tekstbox met de gebruikte simulatieparameters
#ax.text(0.98, 0.98,
#        f'L1 = L2 = {L1*100:.1f} cm\n'
#        f'm1 = {M1*1000:.0f} g,  m2 = {M2*1000:.0f} g\n'
#        f'T = {T_total:.0f} s,  dt = {dt} s',
#        transform=ax.transAxes,   # coördinaten relatief aan de as (0-1)
#        color='white', fontsize=8,
#        va='top', ha='right',
#        bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

# ── Legenda ───────────────────────────────────────────────────────────
# We maken handmatige legenda-symbolen met Line2D omdat scatter-punten
# niet automatisch in de legenda verschijnen via ax.legend()
if len(exp_th1_deg) > 0:
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_CHAOS, markeredgecolor=EDGE_COLOR,
               markeredgewidth=EDGE_WIDTH, markersize=9,
               label='experiment (chaotic)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_NOCHAOS, markeredgecolor=EDGE_COLOR,
               markeredgewidth=EDGE_WIDTH, markersize=9,
               label='experiment (non-chaotic)'),
    ]
    ax.legend(handles=legend_elements,
              loc='upper right', fontsize=17, framealpha=0.55,
              facecolor='black', edgecolor='white', labelcolor='white', markerfirst = False)

# ── Opslaan ───────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)   # automatische marge zodat labels niet worden afgesneden
plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
            facecolor='white', edgecolor='white')
plt.close()   # sluit de figuur om geheugen vrij te maken
print(f"\nFiguur opgeslagen als '{output_file}'")