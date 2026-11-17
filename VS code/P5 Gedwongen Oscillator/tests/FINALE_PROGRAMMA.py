import nidaqmx
import time
import numpy as np
import sympy as sp
import os

# Config
device = "cDAQ1Mod1" #default cDAQ2Mod1
line5 = "port0/line5"   # Oscillator 1
line6 = "port0/line6"   # Oscillator 2
line7 = "port0/line7"   # Amplitude sensor (for oscillator 2)
refresh_rate = 0.000005   # update interval (seconds)


# Start NI-DAQmx Task
#########################################################################
#CORRECT INLEZEN VAN DE DATA
#########################################################################
with nidaqmx.Task() as task:
    task.di_channels.add_di_chan(f"{device}/{line5},{device}/{line6},{device}/{line7}",line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE) # maakt de communicatie met de channels op line 5,6 en 7 van het NI instrument
    try:
        val5_lezingen = []
        val6_lezingen = []
        val7_lezingen = []

        starttijd1, starttijd2 = time.perf_counter(), time.perf_counter()

        val5_tijd = []
        val6_tijd = []

        tijdverschil_fase = []
        tijd_driver_stijging = -1

        vals = task.read(1000)          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals

        val5, val6, val7 = vals
        val5_vorige = val5
        val6_vorige = val6

        while True:
            vals = task.read()          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals
            val5, val6, val7 = vals     # "unpacked" vals in verschillende variabelen
            print(val5,val6,val7)       # print de waarden

            val5_lezingen.append(val5)      # maakt een lijst met True en False-waarden per sensor
            val6_lezingen.append(val6)
            val7_lezingen.append(val7)

            if val5 != val5_vorige:                        #maakt een lijst met tijden op de momenten waar de lijst True of False geeft van sensor 5
                eindtijd1 = time.perf_counter()
                val5_tijd.append(eindtijd1-starttijd1)
                starttijd1 = eindtijd1
                

                if val5 is True: #verandering aandrijver van de fase → wordt positief als verandering zodat het tijdsverschil effectief gemeten wordt
                    tijd_driver_stijging = time.perf_counter()
            
            val5_vorige = val5

            if val6 != val6_vorige:                         #idem voor de lijst van sensor 6
                eindtijd2 = time.perf_counter()
                val6_tijd.append(eindtijd2-starttijd2)
                starttijd2 = eindtijd2

                if val6 is True: #meet tijdsverschil tussen verandering van de fases om het faseverschil te gaan kunnen berekenen
                    if tijd_driver_stijging > 0:
                        tijdsverschil = time.perf_counter() - tijd_driver_stijging
                        tijdverschil_fase.append(tijdsverschil)
                        tijd_driver_stijging = -1
            
                val6_vorige = val6

            time.sleep(refresh_rate)    # wacht gedurende een refreshrate

    except KeyboardInterrupt:
        print("Stopped by user.")

#########################################################################
#BEREKENINGEN MET DE VERKREGEN DATA
#########################################################################

###FREQUENTIE
def freqentie(lijst_tijd):       # berekend de frequentie die er gemeten werdt
    if len(lijst_tijd) <= 1:
        return 0

    lijst_tijd = lijst_tijd[1:] #de eerste tijd zal altijd onrealistisch zijn omdat het programma en de machine nog gestart moeten worden
    gem_tijd = np.mean(lijst_tijd)
    
    if gem_tijd == 0:
        return 0
    
    periode = 2 * gem_tijd
    freqentie = 1 / periode
    return freqentie

freq_5 = freqentie(val5_tijd)
freq_6 = freqentie(val6_tijd)

###AMPLITUDE
lijst_indices =[]
amplitudes_waardes = []
for i in range(len(val6_lezingen)-1): #creeert een lijst met de indices waarop er een wisseling is van True en False in de aandrijver → start en eindpunten van een halve aandrijving
    if val6_lezingen[i] != val6_lezingen[i+1]:
        lijst_indices.append(i)

for i in range(len(lijst_indices)-1): #gebruikt deze lijst om de amplitude te meten adhv de wisselingen
    een_boogje = val7_lezingen[lijst_indices[i]:lijst_indices[i+1]]
    counter = 0
    for i in range(len(een_boogje)-1):
        if een_boogje[i] != een_boogje[i+1]:
            counter += 1
    amplitudes_waardes.append(counter)

if amplitudes_waardes !=[]: #berekend gemiddelde van de amplitudes
    gemiddelde_counter = sum(amplitudes_waardes)/ len(amplitudes_waardes)
    amplitude = gemiddelde_counter / 2
else:
    amplitude = 0

###FASE
fase_rad = 0
if freq_5 > 0 and len(tijdverschil_fase) > 0:
    delta_t_gem = np.mean(tijdverschil_fase)
    periode = 1 / freq_5
    fase_rad = (delta_t_gem / periode) * 2 * np.pi #formule: \varphi = (\Updelta t / T) * 2 \pi
    fase_rad_domein = fase_rad
    if fase_rad_domein > np.pi: #voldoen aan domein [-pi, pi]
        fase_rad_domein = fase_rad_domein - 2 * np.pi


#######################################################################
#FOUTENBEREKENING EN WAARDEN PRINTEN
#######################################################################
def fout_fase(tijd, dtijd, freq, dfreq):
    t, f = sp.symbols('t f')
    phi = (t * f) * 2 * np.pi
    s_phi_func = sp.sqrt((sp.diff(phi, t))**2 * dtijd**2 + sp.diff(phi, f) * dfreq**2)
    s_phi = s_phi_func.subs([(t, tijd), (f, freq)])
    return s_phi

def fout_freq(val_tijd):
    P = 2 * np.mean(val_tijd)
    dP = 2 * np.std(val_tijd)
    T = sp.symbols('T')
    f = 1 / T
    s_f = sp.sqrt((sp.diff(f, T))**2 * dP**2)
    return s_f.subs(T, P)

freq_5_fout = fout_freq(val5_tijd[1:])
freq_6_fout = fout_freq(val6_tijd[1:])
amplitude_fout = np.std(amplitudes_waardes)
tijd_fout = np.std(tijdverschil_fase)
fase_fout = fout_fase(delta_t_gem, tijd_fout, freq_6, freq_6_fout)

print("frequentie motor=", freq_5, "Hz")
print("frequentie oscillator", freq_6, "Hz")
print("amplitude massa=",amplitude, "mm")
print("faseverschil =", fase_rad_domein, "rad")

###PRINTEN OPDAT HET MOOI KOPIEERBAAR IS 
#omega, A, phi, domega, dA, dphi
print(2 * np.pi * freq_6, amplitude, fase_rad_domein,2 * np.pi * freq_6_fout, amplitude_fout, fase_fout)

#######################################################################
# RESULTATEN OPSLAAN IN UNIEK BESTAND
#######################################################################
counter = 1
basename = "meting_resultaten_massa1"
# Zoek naar de eerstvolgende beschikbare bestandsnaam
while os.path.exists(f"{basename}_{counter}.txt"):
    counter += 1
filename = f"{basename}_{counter}.txt"
try:
    with open(filename, 'w') as f:
        f.write(f"{2 * np.pi * freq_6} {amplitude} {fase_rad_domein} {2 * np.pi * freq_6_fout} {amplitude_fout} {fase_fout}\n")
    print(f"Data succesvol opgeslagen in {filename}")
except Exception as e:
    print(f"Kon data niet opslaan: {e}")