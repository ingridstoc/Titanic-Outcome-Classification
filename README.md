## Titanic Outcome Classification

## Description

Setul de date pe care l-am ales conține informații despre persoanele implicate în scufundarea vasului 
Titanic. În ciuda faptului că s-a întâmplat o tragedie, consider că folosind tehnici de inteligență 
artificială, putem găsi o corelație între persoanele care au supraviețuit și caracteristici neașteptate cum 
ar fi clasa la care au cumpărat biletul sau chiar orașul din care s-au îmbarcat pe vas. Aceste corelații, 
ne pot aduce cu un pas mai aproape de a înțelege motivul pentru care a avut loc acest accident, ce au 
avut în plus persoanele care au supraviețuit și cum am putea preveni pe viitor o astfel de catastrofa. 

## Dataset

- Dataset: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Description: The dataset contains passenger information including age, sex, class, and other details, with the goal of predicting survival outcomes.

## Linux Commands Used

```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip3 install matplotlib
pip3 install numpy
pip3 install pandas
pip3 install scikit-learn
```
## Implementation Details
- **Libraries Used**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- **Preprocessing Steps**: 
Feicare tabelă conține 12 coloane cu informații precum: nume, sex, clasa la care a cumpărat biletul, 
de unde s-a îmbarcat, dar cea mai importantă e coloana care ne spune dacă a supraviețuit sau nu 
pasagerul. În setul de date folosit, cea mai mare problemă o reprezintă faptul că datele sunt 
incomplete. Câteva modalități pentru a rezolva această problemă ar fi: 
    + Să elimin în totalitate coloanele atât din setul de training, cât și din cel de test, dacă mai mult 
de jumatate din valori sunt nule. 
    + Să înlocuiesc câmpurile lipsă dintr-o coloană cu o valoare predefinită sau cu valori random. 
Pentru a putea utiliza un model de Machine Learning a trebuit să convertesc valorile de tip text sau 
object în valori numerice. Dar, înainte de a realiza acest lucru, am eliminat coloanele pe care nu le-
am  crezut  relevante  în  procesul  de  clasificare.  Am  eliminat  aceste  coloane  atât  din  setul  de 
antrenare, cât și din cel de test. Acestea sunt: 'Name', 'PassengerId', 'Ticket' și 'Cabin'. Logica pentru 
această decizie a fost următoarea pentru fiecare în parte: 
    +   'Name': consider că numele pe care îl are o persoană nu poate avea un impact real asupra 
eficienței modelului. 
    +   'PassengerId': reprezintă numărul intrării în tabel, cum nu sunt prezentate informații despre 
modul  în  care  au  fost  introduse  persoanele  în  tabelă  am  considerat  că  ordinea  este 
aleatoare, ceea ce nu o face un feature relevant pentru model. 
    +   'Ticket': câmpul ticket este o înșiruire aleatoare de coduri alfa-numerice, nu am observat o 
corelație între codul pe care îl are ticket-ul și clasa sau cabina la care a fost cumpărat biletul. 
Stochirlea Ingrid Ana Maria – 333AA 
 
    +   'Cabin': am eliminat această coloana deoarece în setul de date inițial aproape 3 sferturi din 
valori sunt nule atât pentru setul de antrenare (687/891), cât și pentru cel de testare 
(327/418).   
Am înlocuit pentru fiecare coloană în parte unde erau date lipsă cu valori random predefinite

- **Models Used**: KNeighborsClassifier / Random Forest / GaussianNB
- **Fine-tuning Hyperparameters**: 
Implementarea din Scikit-learn a celor 3 algoritmi de clasificare are mai mulți hiperparametri care 
influențează  rezultatele  finale  ale  modelului  pe  datele  de  testare.  De exemplu, pentru  algoritmul 
KneighborsClassifier, am ales să mă folosesc de parametrii: 'n_neighbors', 'leaf_size', 'weights', 
'algorithm' pentru a găsi cea mai bună combinație dintre pentru model. 
Pentru fiecare hiperparametru am setat un număr diferit de valori. Conform codului de mai jos, există 288 de 
combinații de hiperparametrii. Pentru a găsi setul ideal de parametrii vom folosi tehnica de Random 
Search. Sigur există cel putin un model care să aibă cea mai mare acuratețe posibilă pe setul de test. 
În procesul de random search, sunt selectate în mod aleator combinații diferite de parametrii. Cu 
fiecare alegere pe care o facem, posibilitatea de a găsi acel model cel mai bun crește. Variabila 
confidence reprezintă gradul de siguranță pe care îl avem că aceea combinație de parametrii cu 
acuratațe maximă pe care am găsit-o până acum este cea mai bună posibilă. 

```
grid_knn = {
      'n_neighbors': [2, 3, 5, 7, 10, 12],
      'leaf_size': [15, 20, 30, 40, 50, 60],
      'weights': ["uniform", "distance"],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
```

- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Results and Conclusions

| Model              | Accuracy | Precision | Recall | F1-score | 
|-------------------|----------|------------|--------|----------|
| KNeighborsClassifier | 65% | 53% | 49% | 51% |
| Random Forest      |  93% | 93% | 87% | 90% |
| GaussianNB          | 89% | 80% | 94% | 87% |

În  concluzie,  toate cele 3  modele  sunt  robuste  și  dau  rezultate  bune,  dar  modelul  care  folosește 
Random Forest are o acuratețe mult mai bună, fiind mai potrivit pentru setul meu de date. Modelul care 
utilizează KNeighborsClassifier reușește să realizeze o clasificare corectă în doar 65% din cazuri pe 
datele de testare, iar pe datele de antrenare în 78% din cazuri, ceea ce poate sugera o problemă de 
underfitting. Algoritmul nu este suficient de complex pentru a învața suficiente informații despre 
datele de antrenare, dar are o capacitate de generalizare bună. 

