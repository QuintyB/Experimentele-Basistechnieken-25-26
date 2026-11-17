counter5 = 0
for i in val5_test:
    if i == val5_test[0] and counter5 == 0:
        pass
    if i != val5_test[0]:
        counter5 +=1
    if i == val5_test[0] and counter5 != 0:
        break

tijd_1rot5 = 2 * counter5 * refresh_rate
aandrijffrequentie5 = 1 / tijd_1rot5

print(aandrijffrequentie5)

counter6 = 0
for i in val6_test:
    if i == val6_test[0] and counter6 == 0:
        pass
    if i != val6_test[0]:
        counter6 +=1
    if i == val6_test[0] and counter6 != 0:
        break

tijd_1rot6 = 2 * counter6 * refresh_rate
aandrijffrequentie6 = 1 / tijd_1rot6

print(aandrijffrequentie6)

print(val5_test, val6_test, val7_test)