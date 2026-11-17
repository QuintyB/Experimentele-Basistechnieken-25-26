import nidaqmx
import time
from collections import deque
import numpy as np

# Config
device = "cDAQ1Mod1" #default cDAQ2Mod1
line5 = "port0/line5"   # Oscillator 1
line6 = "port0/line6"   # Oscillator 2
line7 = "port0/line7"   # Amplitude sensor (for oscillator 2)
refresh_rate = 0.0005   # update interval (seconds)

val5_test = []
val6_test = []
val7_test = []

starttijd1, starttijd2 = time.perf_counter(), time.perf_counter()

val5_tijd = np.array([])
val6_tijd = np.array([])

# Start NI-DAQmx Task
with nidaqmx.Task() as task:
    task.di_channels.add_di_chan(f"{device}/{line5},{device}/{line6},{device}/{line7}",line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE) # maakt de communicatie met de channels op line 5,6 en 7 van het NI instrument
    try:
        while True:
            vals = task.read()          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals
            val5, val6, val7 = vals     # "unpacked" vals in verschillende variabelen
            print(val5,val6,val7)       # print de waarden

            val5_test.append(val5)
            val6_test.append(val6)
            val7_test.append(val7)

            if val5 != val5_test[0]:
                eindtijd1 = time.perf_counter()
                val5_tijd.append(eindtijd1-starttijd1)
                starttijd1 = eindtijd1

            if val6 != val6_test[0]:
                eindtijd2 = time.perf_counter()
                val6_tijd.append(eindtijd2-starttijd2)
                starttijd2 = eindtijd2

            time.sleep(refresh_rate)    # wacht gedurende een refreshrate

    except KeyboardInterrupt:
        print("Stopped by user.")

tijd_5_echt = []
for i in val5_tijd:
    if i > refresh_rate + 1e-4:
        tijd_5_echt.append(i)

tijd_6_echt = []
for i in val6_tijd:
    if i > refresh_rate + 1e-4:
        tijd_6_echt.append(i)

tijd_5_echt = tijd_5_echt[1:]
gem_val5 = 2 * (np.sum(tijd_5_echt) / len(tijd_5_echt))

freq_5 = 1 / gem_val5

tijd_6_echt = tijd_6_echt[1:]
gem_val5 = 2 * (np.sum(tijd_6_echt) / len(tijd_6_echt))
freq_5 = 1 / gem_val5