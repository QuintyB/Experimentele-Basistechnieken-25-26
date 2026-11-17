import random
welke_oefenen = input("Wat wil je oefenen? Tripeptiden (T) of Aminozuren (A):")
aminozuren = ['glycine', 'alanine', 'valine', 'leucine', 'isoleucine', 'serine', 'threonine', 'aspartaanzuur', 'asparagine', 'glutaminezuur', 'glutamine', 'phenylalanine', 'cysteïne', 'methionine', 'lysine', 'arginine', 'tyrosine', 'histedine', 'tryptophan', 'proline']
letters = ['G', 'A', 'V', 'L', 'I', 'S', 'T', 'D', 'N', 'E', 'Q', 'F', 'C', 'M', 'K', 'R', 'Y', 'H', 'W', 'P']
pH = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
if welke_oefenen.lower() == 't':
    random.shuffle(letters)
    random.shuffle(pH)
    tripep = letters[0] + letters[1] + letters[2]
    pH_val = pH[0]
    print(tripep, "bij pH", pH_val)
else:
    random.shuffle(aminozuren)
    for i in aminozuren:
        print(i)
        