# Autoencoder_
## EDITEURS:

**LUCEA** **LENNY**

**VALDEYRON** **MATHIEU**

**THIRIET** **AURÉLIEN**

## Applications python autoencoders

L'objectif de ces applications est de montrer dans quel cadre nous pouvons utiliser les autoencoders et en quoi ils peuvent s'avérer utiles. 

Pour ce faire nous utiliserons les packages python torch et keras que nous appliquerons à notre base de donnée tirée de MNIST.

## I. Autoencoder pour la classification


Dans un premier temps, nous comparerons les résultats de classification des autoencodeurs à celle d'un autre outil de classification. Ici nous avons choisi la forêt aléatoire. Cela nous permettra de comparer leurs MSE sur un même jeux de données !

<img src="https://github.com/lucea97217/Autoencoder_/blob/main/digit.png" >

Nous observerons que les résultats donnés par la méthode de forêt aléatoire s'avère moins concluants que ceux donnés par la méthode autoencoder.

## II. Denoising autoencoder (DAE)

Dans un second temps nous introduirons du bruit à nos données afin d'observer si la méthode DAE nous permet de retomber sur nos données d'origine.


<img src="https://github.com/lucea97217/Autoencoder_/blob/main/Capture%20d’écran%202022-10-14%20à%2020.57.00.png" >

On retrouve des résultats très concluants !


Sources : <url="https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e">

<url="https://www.kaggle.com/code/stephanedc/tutorial-keras-autoencoders">