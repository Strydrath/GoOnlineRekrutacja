# GoOnline Rekrutacja

## Opis Wyników

Ze względu na charakterystykę danych: jednakowe tło, podobne oświetlenie, brak elementów innych niż dłoń na zdjęciach a także fakt, że wszystkie zdjęcia z klasy 0 były jednakowe - samo tło, klasyfikacja była łatwa do przeprowadzenia i uzyskano wynik bliski 100% dokładności. 

Dane były również na tyle do siebie podobne, że mało prawdopodobnym było zjawisko przeuczenia, dodatkowo jego ryzyko zostało zmniejszone poprzez dodanie warstwy Dropout.

### Krzywe uczenia
![image](https://github.com/Strydrath/GoOnlineRekrutacja/assets/40769763/0e543d20-84f1-4c23-b2c9-583580cd3209)
![image](https://github.com/Strydrath/GoOnlineRekrutacja/assets/40769763/a4f0f8d8-a943-45b7-878d-cc8df4384530)

Wyświetlone krzywe uczenia pokazują, że już około epoki 15 model został wyuczony, następnie wyniki utrzymywały się na tym podobnym poziomie.

Można zauważyć również, że metryki są wyższe dla danych walidacyjnych niż dla danych treningowych. Może się to wiązać z mniejszym zestawem danych oraz mniejszą szansą na zastosowanie obrotu bądź odbicia.

### Ewaluacja
Ewaluacja została przeprowadzona przy pomocy biblioteki scikit-learn. Obliczone zostały metryki accuracy, precision, recall oraz F1-score.
Predykcja na danych testowych wykazała wynik 100% dokładności. Każda z metryk osiągnęła wartość 1, ze względu na pełną trafność predykcji.

### Wyniki predykcji losowych zdjęć:

![image](https://github.com/Strydrath/GoOnlineRekrutacja/assets/40769763/ca271e1f-6685-4626-bb18-a1e99ad292a0)

![image](https://github.com/Strydrath/GoOnlineRekrutacja/assets/40769763/2246fd47-c91a-4b00-a9a9-ec4e5733cefa)

![image](https://github.com/Strydrath/GoOnlineRekrutacja/assets/40769763/b53ad303-e68c-4238-b486-9f0b598acfc0)
