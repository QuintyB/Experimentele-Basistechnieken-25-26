"""
╔══════════════════════════════════════════════════════════════════════╗
║   LYAPUNOV EXPONENT PUUR UIT EXPERIMENTELE DATA (Rosenstein 1993)   ║
║   Geen simulatie gebruikt — chaos bepaald uit de gemeten tijdreeks   ║
╚══════════════════════════════════════════════════════════════════════╝

HOE WERKT DIT?
──────────────
Voor elke proef:
  1. Lees θ₁(t) en θ₂(t) uit de Tracker-bestanden
  2. Schat ω₁(t) en ω₂(t) numeriek (afgeleide)
  3. Bouw een 4D-toestandsvector per tijdstip: [θ₁, ω₁, θ₂, ω₂]
  4. Zoek voor elk tijdstip de dichtstbijzijnde buur in de toestandsruimte
     (maar niet een buur die vlak erbij ligt in de tijd)
  5. Volg hoe ver die buren uit elkaar groeien → log(afstand) vs tijd
  6. De helling van die curve = Lyapunov exponent λ [1/s]
     λ > 0  →  chaotisch
     λ ≤ 0  →  stabiel / periodiek

VERGELIJKING MET THEORIE:
  De theoretische chaos-kaart (uit eigen_dp_fractaal_met_punten.py)
  voorspelt of een beginpunt chaotisch is op basis van simulatie.
  Dit script bepaalt hetzelfde ONAFHANKELIJK uit het experiment zelf.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os


# ══════════════════════════════════════════════════════════════════════
# PARAMETERS  ← zelfde als in eigen_dp_fractaal_met_punten.py
# ══════════════════════════════════════════════════════════════════════

BASE      = '/Users/swinn/Documents/2de bachelor/experimentel basistechnieken/VS code/vrij proef'
N_PROEVEN = 32
output_file = 'lyapunov_experimenteel.png'
dpi         = 220

# Rosenstein-parameters
MIN_TIJDSAFSTAND = 10   # minimale tijdindex-afstand tussen een punt en zijn buur
                        # voorkomt dat buren worden gekozen die gewoon "de volgende
                        # meting" zijn (temporele autocorrelatie vermijden)
                        # vuistregel: stel in als ~halve periode van de slinger
MAX_STAPPEN      = 40   # hoeveel tijdstappen we de divergentie volgen
                        # te groot → signaal wordt ruis; te klein → onbetrouwbaar
                        # kies zo dat MAX_STAPPEN * dt ≈ 1–2 s

# Drempelwaarden
THRESHOLD_LAMBDA = 0.0   # λ > 0 → chaotisch (de klassieke definitie)
                          # je kunt dit ook op 0.20 zetten om consistent
                          # te zijn met de chaos-kaart

# Kleuren (zelfde stijl als de chaos-kaart)
COLOR_CHAOS   = '#00FFEE'   # cyaan   → chaotisch
COLOR_NOCHAOS = '#39FF14'   # groen   → niet-chaotisch
COLOR_FOUT    = '#FF4444'   # rood    → te weinig data / berekening mislukt


# ══════════════════════════════════════════════════════════════════════
# HULPFUNCTIES (zelfde als in het originele script)
# ══════════════════════════════════════════════════════════════════════

def lees_tracker(pad, ncols=3):
    """
    Leest een Tracker-exportbestand in.
    Verwacht: 2 headerregels, tab-gescheiden, komma als decimaalteken.
    Geeft een numpy-array terug met vorm (n_frames, ncols).
    """
    rijen = []
    with open(pad, encoding='utf-8') as f:
        for nr, lijn in enumerate(f):
            if nr < 2:
                continue
            lijn = lijn.strip()
            if not lijn:
                continue
            delen = lijn.replace(',', '.').split('\t')
            try:
                rijen.append([float(delen[j]) for j in range(ncols)])
            except (ValueError, IndexError):
                pass
    return np.array(rijen)


def wrap(theta):
    """Vouwt een hoek terug naar [-π, π]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# ══════════════════════════════════════════════════════════════════════
# KERN: ROSENSTEIN-METHODE
# ══════════════════════════════════════════════════════════════════════

def rosenstein_lyapunov(toestand, dt, min_tijdsafstand=MIN_TIJDSAFSTAND,
                        max_stappen=MAX_STAPPEN):
    """
    Berekent de maximale Lyapunov exponent uit een experimentele tijdreeks
    via de methode van Rosenstein et al. (1993).

    Invoer:
      toestand        → array (n_frames, n_dim): toestandsvector per tijdstip
                        in ons geval: [θ₁, ω₁, θ₂, ω₂] per frame
      dt              → tijdstap tussen frames [s]
      min_tijdsafstand→ minimale tijdindex-afstand tussen punt en buur
      max_stappen     → hoe ver we de divergentie volgen

    Uitvoer:
      lamba           → geschatte Lyapunov exponent [1/s]
      tijden          → tijdas van de divergentiecurve [s]
      gem_log_afstand → gemiddelde log(afstand) als functie van tijd
                        (de curve waarvan we de helling nemen)
    """
    n = len(toestand)

    # ── Stap 1: zoek voor elk tijdstip de dichtstbijzijnde buur ──────
    # KDTree maakt een efficiënte zoekstructuur over alle toestandsvectoren
    boom = KDTree(toestand)

    # query(..., k=2) geeft de 2 dichtstbijzijnde buren terug:
    #   resultaat[0] = het punt zelf (afstand 0)
    #   resultaat[1] = de echte dichtstbijzijnde buur
    _, alle_buren = boom.query(toestand, k=min_tijdsafstand + 2)

    # Kies per punt de dichtstbijzijnde buur die minstens min_tijdsafstand
    # tijdstappen verwijderd is (om temporele buren te vermijden)
    buur_idx = np.full(n, -1, dtype=int)
    for t in range(n):
        for kandidaat in alle_buren[t, 1:]:   # skip zichzelf (index 0)
            if abs(int(kandidaat) - t) >= min_tijdsafstand:
                buur_idx[t] = int(kandidaat)
                break

    # Punten waarvoor geen geschikte buur gevonden werd, laten we weg
    geldig = buur_idx >= 0

    # ── Stap 2: volg de divergentie over max_stappen tijdstappen ─────
    # Voor elk geldig beginpunt t berekenen we na k stappen:
    #   afstand(t+k) = ||toestand[t+k] - toestand[buur[t]+k]||
    # en nemen we log(afstand)
    log_afstanden = []   # elke rij = log(afstand) over de tijd voor één beginpunt

    for t in np.where(geldig)[0]:
        b = buur_idx[t]
        # Zorg dat we niet buiten de array gaan
        beschikbaar = min(max_stappen, n - t - 1, n - b - 1)
        if beschikbaar < 5:   # te weinig stappen → overslaan
            continue

        rij = []
        for k in range(beschikbaar):
            d = np.linalg.norm(toestand[t + k] - toestand[b + k])
            d = max(d, 1e-15)   # vermijd log(0)
            rij.append(np.log(d))
        log_afstanden.append(rij)

    if len(log_afstanden) == 0:
        return None, None, None   # te weinig data

    # ── Stap 3: middel over alle beginpunten ─────────────────────────
    # Gebruik alleen de kortste gemeenschappelijke lengte
    min_len = min(len(r) for r in log_afstanden)
    matrix  = np.array([r[:min_len] for r in log_afstanden])
    gem_log_afstand = np.mean(matrix, axis=0)   # vorm (min_len,)

    tijden = np.arange(min_len) * dt   # tijdas in seconden

    # ── Stap 4: lineaire fit → helling = λ ───────────────────────────
    # De Lyapunov exponent is de helling van de rechte lijn door
    # gem_log_afstand(t). We fitten op het lineaire deel (eerste helft).
    fit_einde = max(5, min_len // 2)
    coeffs    = np.polyfit(tijden[:fit_einde], gem_log_afstand[:fit_einde], 1)
    lam       = coeffs[0]   # helling [1/s]

    return lam, tijden, gem_log_afstand


# ══════════════════════════════════════════════════════════════════════
# HOOFDLUS: VERWERK ALLE PROEVEN
# ══════════════════════════════════════════════════════════════════════

resultaten = []   # lijst van dicts met resultaat per proef

print("Lyapunov exponent berekenen uit experimentele data ...\n")

for i in range(1, N_PROEVEN + 1):
    pad_a = f'{BASE}/data/data proef twee/proef_2_{i}_A.txt'
    pad_b = f'{BASE}/data/data proef twee/proef_2_{i}_B.txt'

    if not (os.path.exists(pad_a) and os.path.exists(pad_b)):
        continue

    # ── Data inlezen ──────────────────────────────────────────────────
    raw_b = lees_tracker(pad_b)
    raw_a = lees_tracker(pad_a)
    n     = min(len(raw_b), len(raw_a), 600)

    if n < 50:   # te weinig frames voor een betrouwbare berekening
        print(f"  Proef {i}: OVERGESLAGEN (slechts {n} frames)")
        continue

    t_arr = raw_b[:n, 0]
    xB, yB = raw_b[:n, 1], raw_b[:n, 2]
    xA, yA = raw_a[:n, 1], raw_a[:n, 2]

    # ── Hoeken berekenen (volledige tijdreeks) ────────────────────────
    # arctan2(x, -y) → hoek t.o.v. neerwaartse richting
    # unwrap → verwijder sprongen van -π naar +π
    # wrap   → breng terug naar [-π, π]
    th1 = wrap(np.unwrap(np.arctan2(xB, -yB)))
    th2 = wrap(np.unwrap(np.arctan2(xA - xB, yB - yA)))

    # ── Tijdstap bepalen ──────────────────────────────────────────────
    # Gebruik de gemiddelde tijdstap uit de Tracker-data
    dt_meting = float(np.mean(np.diff(t_arr)))

    # ── Hoeksnelheden numeriek schatten ──────────────────────────────
    # np.gradient gebruikt centrale differenties (nauwkeuriger dan simpel verschil)
    om1 = np.gradient(th1, t_arr)
    om2 = np.gradient(th2, t_arr)

    # ── Toestandsvector bouwen: [θ₁, ω₁, θ₂, ω₂] per tijdstip ───────
    toestand = np.column_stack([th1, om1, th2, om2])

    # ── Rosenstein-methode toepassen ──────────────────────────────────
    lam, tijden, gem_log_d = rosenstein_lyapunov(
        toestand, dt_meting,
        min_tijdsafstand=MIN_TIJDSAFSTAND,
        max_stappen=MAX_STAPPEN
    )

    if lam is None:
        print(f"  Proef {i}: MISLUKT (te weinig bruikbare buren)")
        resultaten.append({'proef': i, 'lam': None,
                           'th1_0': np.degrees(th1[0]),
                           'th2_0': np.degrees(th2[0]),
                           'tijden': None, 'gem_log_d': None})
        continue

    chaotisch = lam > THRESHOLD_LAMBDA
    print(f"  Proef {i:2d}: λ = {lam:+.3f} s⁻¹  "
          f"th1={np.degrees(th1[0]):+.1f}°  th2={np.degrees(th2[0]):+.1f}°  "
          f"→  {'CHAOTISCH' if chaotisch else 'STABIEL'}")

    resultaten.append({
        'proef'    : i,
        'lam'      : lam,
        'chaotisch': chaotisch,
        'th1_0'    : np.degrees(th1[0]),
        'th2_0'    : np.degrees(th2[0]),
        'tijden'   : tijden,
        'gem_log_d': gem_log_d,
    })


# ══════════════════════════════════════════════════════════════════════
# FIGUUR 1 — Divergentiecurves per proef
# ══════════════════════════════════════════════════════════════════════
# Elke curve toont log(gemiddelde afstand) vs tijd.
# Een rechte stijgende lijn = exponentiële divergentie = chaos.
# De helling van die lijn = λ.

geldig_res = [r for r in resultaten if r['lam'] is not None]
n_plots    = len(geldig_res)

if n_plots > 0:
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols
    fig, assen = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 2.8),
                              facecolor='#0d0d0d')
    assen_flat = assen.flatten() if n_plots > 1 else [assen]

    for ax, r in zip(assen_flat, geldig_res):
        kleur = COLOR_CHAOS if r['chaotisch'] else COLOR_NOCHAOS
        ax.set_facecolor('#1a1a1a')

        # Plot de divergentiecurve
        ax.plot(r['tijden'], r['gem_log_d'], color=kleur, linewidth=1.4)

        # Teken de gefitte rechte lijn over het eerste deel
        fit_einde = max(5, len(r['tijden']) // 2)
        t_fit  = r['tijden'][:fit_einde]
        coeffs = np.polyfit(t_fit, r['gem_log_d'][:fit_einde], 1)
        ax.plot(t_fit, np.polyval(coeffs, t_fit),
                color='white', linewidth=0.9, linestyle='--', alpha=0.7)

        ax.set_title(f"Proef {r['proef']}   λ={r['lam']:+.2f} s⁻¹",
                     color=kleur, fontsize=8, pad=3)
        ax.set_xlabel('tijd [s]', color='#888888', fontsize=7)
        ax.set_ylabel('ln(afstand)', color='#888888', fontsize=7)
        ax.tick_params(colors='#666666', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # Verberg lege subplots
    for ax in assen_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Divergentiecurves — Rosenstein-methode\n'
                 'stijgende lijn = divergentie = chaos  |  '
                 'hellingshoek = λ',
                 color='white', fontsize=11, y=1.01)
    plt.tight_layout(pad=0.8)
    plt.savefig('lyapunov_curves.png', dpi=dpi, bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print("\nDivergentiecurves opgeslagen als 'lyapunov_curves.png'")


# ══════════════════════════════════════════════════════════════════════
# FIGUUR 2 — Overzichtsplot: λ per proef als staafdiagram
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(max(8, n_plots * 0.5), 5), facecolor='#0d0d0d')
ax.set_facecolor('#1a1a1a')

proef_nrs = []
lambda_vals = []
kleuren_bars = []

for r in resultaten:
    proef_nrs.append(r['proef'])
    if r['lam'] is None:
        lambda_vals.append(0)
        kleuren_bars.append(COLOR_FOUT)
    else:
        lambda_vals.append(r['lam'])
        kleuren_bars.append(COLOR_CHAOS if r['chaotisch'] else COLOR_NOCHAOS)

bars = ax.bar(proef_nrs, lambda_vals, color=kleuren_bars,
              edgecolor='#333333', linewidth=0.5, width=0.7)

# Drempellijn bij λ = 0
ax.axhline(0, color='white', linewidth=1.0, linestyle='--', alpha=0.6,
           label='λ = 0  (grens chaos)')

ax.set_xlabel('Proefnummer', color='white', fontsize=11)
ax.set_ylabel('Lyapunov exponent  λ  [1/s]', color='white', fontsize=11)
ax.set_title('Experimentele Lyapunov exponent per proef\n'
             '(Rosenstein-methode — geen simulatie gebruikt)',
             color='white', fontsize=12)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#555555')

# Legenda
from matplotlib.patches import Patch
legenda = [
    Patch(facecolor=COLOR_CHAOS,   edgecolor='white', label='Chaotisch  (λ > 0)'),
    Patch(facecolor=COLOR_NOCHAOS, edgecolor='white', label='Stabiel    (λ ≤ 0)'),
    Patch(facecolor=COLOR_FOUT,    edgecolor='white', label='Onvoldoende data'),
]
ax.legend(handles=legenda, facecolor='#1a1a1a', edgecolor='white',
          labelcolor='white', fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Overzichtsplot opgeslagen als '{output_file}'")


# ══════════════════════════════════════════════════════════════════════
# TEKSTSAMENVATTING
# ══════════════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("SAMENVATTING")
print("═" * 55)
n_chaotisch = sum(1 for r in resultaten if r.get('chaotisch', False))
n_stabiel   = sum(1 for r in resultaten if r.get('lam') is not None
                  and not r['chaotisch'])
n_mislukt   = sum(1 for r in resultaten if r['lam'] is None)
print(f"  Chaotisch  (λ > 0) : {n_chaotisch} proeven")
print(f"  Stabiel    (λ ≤ 0) : {n_stabiel} proeven")
print(f"  Onvoldoende data   : {n_mislukt} proeven")
print("═" * 55)
print("\nDit resultaat is ONAFHANKELIJK van de simulatie.")
print("Je kunt het nu vergelijken met de chaos-kaart:\n"
      "  → Zitten de chaotische proeven ook in de gele/rode zones?")