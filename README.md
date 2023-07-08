# GoOnline Rekrutacja
## Dokumentacja

Dokumentacja funkcji znajduje się w pliku z kodem w formie docstringów

Podpunkty zadania:
1. Wczytanie danych:
  - load_data - załadowanie danych z folderów do obiektów typu Dataset oraz podział na zbiory treningowy, walidacyjny oraz testowy, dane podzielone są na batche każdy po 16 próbek
  - load_labels - wczytanie etykiet z nazw folderów
2. Normalizacja danych:
  - w funkcji prepare_model zostaje dodana warstwa Rescaling(1. / 255)
3. Augmentacja danych:
  - w funkcji prepare_model dodane zostają warstwy:
  *  RandomFlip - losowe odbicia zdjęć
  *  RandomRotation - losowe obrócenie zdjęć
4. Stworzyć własny model:
  - prepare_model - stworzony zostaje model z warstwami:
    * Conv2D(8, 3)
    * MaxPooling2D()
    * Conv2D(16, 3)
    * MaxPooling2D()
    * Conv2D(32, 3)
    * MaxPooling2D()
    * Conv2D(32, 3)
    * MaxPooling2D()
    * Dropout(0.3)
    * Flatten()
    * Dense(128)
    * Dense(4)
5. Skompilować model:
  - main - model zostaje skompilowany z:
  - optymalizatorem RMSprop(learning_rate=0.0001),
  - funkcją straty SparseCategoricalCrossentropy(from_logits=True)
  - metryką accuracy
6. Wytrenować model:
  - train_model - model zostaje wytrenowany podczas 18 epok, w każdej epoce użyte są wszystkie dane ze zbioru treningowego - wykorzystane jest 60 batchy
7. Zapisać model do pliku:
  - main - model zapisany jest do pliku model.keras
8. Wykreślić krzywe uczenia:
  - plot_learning_curves - z obiektu history wczytane są dane o procesie uczenia oraz wykreślone wykresy accuracy oraz loss
9. Przetestwać model:
  - evaluate_model - wykonana jest predykcja dla wszystkich batchy zbioru testowego i przy pomocy biblioteki scikit-learn obliczone są metryki accuracy, precision, recall oraz f1-score
10. Pokazać wynik predykcji dla 10 losowych obrazów
  - show_random_predictions - zbiór testowy zostej przetasowany i zostaje wykonana predykcja na pierwszych 10 elementach zbioru oraz pokazany wynik przy pomocy biblioteki matplotlib

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
