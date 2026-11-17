import nidaqmx
import time
import numpy as np

# Config
device = "cDAQ1Mod1" #default cDAQ2Mod1
line5 = "port0/line5"   # Oscillator 1
line6 = "port0/line6"   # Oscillator 2
line7 = "port0/line7"   # Amplitude sensor (for oscillator 2)
refresh_rate = 0.0005   # update interval (seconds)


# Start NI-DAQmx Task
with nidaqmx.Task() as task:
    task.di_channels.add_di_chan(f"{device}/{line5},{device}/{line6},{device}/{line7}",line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE) # maakt de communicatie met de channels op line 5,6 en 7 van het NI instrument
    try:
        val5_test = []
        val6_test = []
        val7_test = []

        starttijd1, starttijd2 = time.perf_counter(), time.perf_counter()

        val5_tijd = []
        val6_tijd = []
        vals = task.read()          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals
        val5, val6, val7 = vals
        val5_vorige = val5
        val6_vorige = val6
        while True:
            vals = task.read()          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals
            val5, val6, val7 = vals     # "unpacked" vals in verschillende variabelen
            #print(val5,val6,val7)       # print de waarden

            val5_test.append(val5)      # maakt een lijst met True en false per sensor
            val6_test.append(val6)
            val7_test.append(val7)

            if val5 != val5_vorige:                        #maakt een lijst met tijden op de momenten waar de lijst true of false geeft
                eindtijd1 = time.perf_counter()
                val5_tijd.append(eindtijd1-starttijd1)
                starttijd1 = eindtijd1
                val5_vorige = val5

            if val6 != val6_vorige:                         # idem voor de lijst van 6
                eindtijd2 = time.perf_counter()
                val6_tijd.append(eindtijd2-starttijd2)
                starttijd2 = eindtijd2
                val6_vorige = val6

            



            time.sleep(refresh_rate)    # wacht gedurende een refreshrate

    except KeyboardInterrupt:
        print("Stopped by user.")

def freqentie(lijst_tijd):       # berekend de frequentie die er gemeten werdt
    if len(lijst_tijd) <= 1:
        return 0

    lijst_tijd = lijst_tijd[1:]
    gem_tijd = np.mean(lijst_tijd)
    
    if gem_tijd == 0:
        return 0
    
    periode = 2 * gem_tijd
    freqentie = 1 / periode
    return freqentie


freq_5 = freqentie(val5_tijd)
freq_6 = freqentie(val6_tijd)

l_m_i =[]
LL = []
for i in range(len(val6_test)-1):       
    if val6_test[i] != val6_test[i+1]:
        index = i 
        l_m_i.append(i)
for i in range(len(l_m_i)-1):
    een_boogje = val7_test[l_m_i[i]:l_m_i[i+1]]
    counter = 0
    for i in range(len(een_boogje)-1):
        if een_boogje[i] != een_boogje[i+1]:
            counter += 1
    LL.append(counter)
if LL !=[]:
    gemiddelde_counter = sum(LL)/ len(LL)
    amplitude = gemiddelde_counter / 2
else:
    amplitude = 0


print("frequentie motor=", freq_5)
print("frequentie oscilartor", freq_6)
print("amplitude=",amplitude)
