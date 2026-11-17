import nidaqmx
import time

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
        while True:
            vals = task.read()          # Leest de huidige waarde uit van line 5,6 en 7 en slaagt deze op in vals
            val5, val6, val7 = vals     # "unpacked" vals in verschillende variabelen
            print(val5,val6,val7)       # print de waarden

            time.sleep(refresh_rate)    # wacht gedurende een refreshrate

    except KeyboardInterrupt:
        print("Stopped by user.")