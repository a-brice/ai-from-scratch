
<h1>Coloration des graphes</h1>

<h4>Introduction :</h4>

Le projet que nous avons dû réaliser en binôme avait pour objet la coloration de graphes avec 2 ou 3 couleurs. Nous avons choisi d’utiliser deux approches, l’exponentielle et la polynomiale, afin de pouvoir clairement comparer leurs efficacités respectives, et de pouvoir se faire une idée concrète des différences entre ces deux types d’algorithmes. Nous avons également réalisé une interface graphique, afin d’avoir une vision plus claire du résultat donné par nos algorithmes, et nous avons implémenté un timer, afin de pouvoir comparer la vitesse de résolution entre nos différents algorithmes.
 
Choix de technologie : 

La première technologie à laquelle nous avons pensé est Python, car la richesse de ses bibliothèques et à l’abondance de ressources en ligne constituait des arguments solides en sa faveur, mais le manque de connaissances de ce langage nous a fait nous orienter vers C#.
Nous avons choisi d’utiliser C# pour réaliser notre projet, étant donné que nous le maîtrisions tous les deux, et que par conséquent nous n’avions pas besoin de nous former à une nouvelle technologie afin d’entamer le projet.

Classes du programme :
Notre programme se divise en deux classes : une classe Maillon, qui instancie chaque point de la figure, et une classe Lien, qui représente les liens entre les différents points de la figure. A l’aide de ces deux classes, il nous est possible de représenter chaque configuration de figure possible.

<h3>Algorithme polynomial :</h3> 
L’algorithme polynomial de notre projet fonctionne de la façon suivante : 
Il prend en paramètre le nombre de maillons, les liens entre eux ainsi que la palette de couleur utilisée, puis il attribue au premier maillon une couleur.
Ensuite, l’algorithme va tour à tour isoler chaque maillon, et une fonction va se charger de déterminer quels maillons sont liés au maillon m étudié. Grâce à cette information, on va pouvoir, à l’aide d’une autre fonction, déterminer quelles couleurs sont applicables à m.


 ![image](https://user-images.githubusercontent.com/78383419/109856546-e4f30f00-7c59-11eb-9d7e-8810d844c337.png)

Fonction déterminant les maillons liés au maillon « som »

 ![image](https://user-images.githubusercontent.com/78383419/109856576-f0ded100-7c59-11eb-9cd1-a227f47ca6c6.png)

Fonction déterminant les couleurs applicables au maillon « som » en fonction des couleurs des maillons liés à lui

Enfin, une autre fonction se charge de colorier chaque maillon. Une particularité de cette fonction est que si dans une certaine configuration de couleurs, il est impossible de colorier un maillon, l’algorithme cherchera à modifier la couleur du maillon précédent (tout en le gardant valable dans le modèle) afin de déboucher la situation. Cependant, si aucune solution satisfaisante n’est trouvée, l’algorithme colore le maillon en noir, et passe au maillon suivant.

Nous avons poussé cet algorithme jusqu’à pouvoir gérer une coloration selon une palette de 4 couleurs, afin de pouvoir colorier un maximum de graphiques.

Tests de l’algorithme polynomial
Test 4 Maillons 6 Liens 3 Couleurs :

<p align="center">
<img src="https://user-images.githubusercontent.com/78383419/109856615-fd632980-7c59-11eb-86a4-779d964047bc.png" >
<img src="https://user-images.githubusercontent.com/78383419/109856650-0653fb00-7c5a-11eb-81f0-77c7459920ca.png">
</p>
  
Nous pouvons constater que le maillon 4 est noir : le programme n’a pas trouvé de configuration dans laquelle ce maillon peut être coloré.
Cependant, si nous faisons le même test avec cette fois 4 couleurs :

<p align="center">
<img src="https://user-images.githubusercontent.com/78383419/109856698-1370ea00-7c5a-11eb-94e6-3c0d8882b151.png">
<img src="https://user-images.githubusercontent.com/78383419/109856716-18ce3480-7c5a-11eb-9117-fdfd2d55c6d7.png">
</p>

Cette fois, le programme a réussi à tout colorer. Nous pouvons également noter que le passage d’une palette de 3 couleurs à une palette de 4 couleurs n’a pas significativement augmenté le temps d’exécution du programme, preuve que ce dernier n’est pas sensible à la taille de la palette.
En théorie, ce programme est capable de colorer des graphes de très grandes tailles.
Test 15 maillons 20 liens 4 couleurs :
 
 <img src="https://user-images.githubusercontent.com/78383419/109856752-22f03300-7c5a-11eb-8d02-7d1f7f286a86.png" width="500">

20 maillons 30 liens 4 couleurs : 

 <img src="https://user-images.githubusercontent.com/78383419/109856777-2a174100-7c5a-11eb-9747-a9a974e4635d.png" width="500">

25 maillons 35 liens 4 couleurs : 

 <img src="https://user-images.githubusercontent.com/78383419/109856806-326f7c00-7c5a-11eb-90c2-ee8c133c9666.png" width="500">

Nous pouvons constater que le temps d’exécution augmente de façon très raisonnable.

Cependant, à partir de 30 maillons et 40 liens, le programme tourne à l’infini, sans pour autant crasher.


<h3>Algorithme exponentiel : </h3>
 
L’algorithme exponentiel de notre projet fonctionne plus simplement :
Tout d’abord, une fonction est chargée de calculer toutes les configurations possibles de colorisation du graphe, à l’aide d’un tableau de (nombreDeMaillons) colonnes et de (nombreDeMaillons^nombreDeCouleurs) lignes.


<img src="https://user-images.githubusercontent.com/78383419/109856919-5af77600-7c5a-11eb-901c-81d34a76ea37.png"  width="500">


Ensuite, une autre fonction se charge d’éliminer les configurations invalides (présentant deux maillons de la même couleur reliés ensembles), afin de ne garder qu’une liste contenant toutes les configurations possibles.
Cet algorithme est également capable d’utiliser jusqu’à 5 couleurs afin de colorer des graphes.

 ![image](https://user-images.githubusercontent.com/78383419/109856995-719dcd00-7c5a-11eb-9c81-09cb260c2267.png)

L’interface graphique liée à cet algorithme nous permet également de visualiser toutes les configurations fonctionnelles pour une figure, contrairement à l’algorithme polynomial.
Test de l’algorithme exponentiel : 
Test 4 maillons 6 liens 4 couleurs : 
 ![image](https://user-images.githubusercontent.com/78383419/109857064-87ab8d80-7c5a-11eb-9c05-a0ded5949949.png)

Ce test permet de confirmer que lorsqu’un faible nombre de points et de liens sont utilisés, l’algorithme exponentiel est déjà moins efficace en termes de temps que l’algorithme polynomial.
Le temps d’exécution pour cette configuration reste quand même raisonnable.
Cependant, lorsque le nombre de points augmente, nous pouvons constater que l’algorithme exponentiel est beaucoup moins performant que son homologue polynomial.

 ![image](https://user-images.githubusercontent.com/78383419/109857121-9eea7b00-7c5a-11eb-9db5-c64dc32edc5a.png)

Pour un graphe ne comprenant que 10 points, l’algorithme n’est déjà plus fonctionnel.
Pour ce qui est de la palette de couleurs utilisées, nous pouvons logiquement conclure que plus la palette comprend de couleurs, et plus l’algorithme mettra de temps à s’exécuter.

Ainsi, pour un test 7 maillons 10 liens 4 couleurs, nous obtenons un résultat, alors que pour un test 7 maillons 10 liens 5 couleurs, le programme crash.
L’ajout d’une couleur dans la palette peut donc représenter un élément critique pour le fonctionnement de cet algorithme.
De plus, le temps d’exécution pour 7 maillons 10 liens devient beaucoup plus long en exponentiel qu’en polynomial (rapport de x10)
  ![image](https://user-images.githubusercontent.com/78383419/109857198-baee1c80-7c5a-11eb-816f-e528ee201407.png)
![image](https://user-images.githubusercontent.com/78383419/109857220-c17c9400-7c5a-11eb-88aa-8828b581065e.png)


<h4>Algorithme vérificateur :</h4>

Cet algorithme prend en paramètre un tableau de Liens et la couleur des maillons, entrés par l’utilisateur.
Ensuite, il vérifie si chaque maillon de chaque lien de la figure a bien une couleur différente : si oui, il indique que la figure est correcte, sinon il indique qu’elle est incorrecte. Dans tous les cas, la figure est ensuite affichée telle que fournie par l’utilisateur.
 ![image](https://user-images.githubusercontent.com/78383419/109857330-dbb67200-7c5a-11eb-824e-c7335ddfe807.png)
![image](https://user-images.githubusercontent.com/78383419/109857374-e7099d80-7c5a-11eb-909b-92cdee53d8f9.png)

 
<h4>Conclusion et pistes d’améliorations :</h4> 
En conclusion, l’algorithme exponentiel est évidemment beaucoup moins efficace que l’algorithme polynomial. En effet, ce dernier est capable de calculer un graphe correct dans une configuration avec beaucoup plus de points et de liens que l’exponentiel, puisque le nombre de calculs à effectuer devient trop important très rapidement.

Pour les pistes d’amélioration, nous pouvons en noter 2 principales : 
-	une amélioration de l’algorithme polynomial, qui est censé fonctionner pour un nombre potentiellement infini de points, mais qui pour nous n’est pas capable de gérer plus de 30 points.

-	Une amélioration des liens générés entre les différents maillons. En effet, il n’était pas rare, lors de nos simulations, de tomber sur des graphes dans lesquels un point ou plus n’étaient pas reliés au reste de la figure, lorsque le nombre de liens n’est pas beaucoup plus important que le nombre de maillons.
