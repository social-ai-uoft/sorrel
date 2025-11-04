# How to populate the environmet?

## *resource population*

We will randomly sample X tuples (latent 1, latent 2, ...) to generate the resource icons. 

## *definition of similarity*

There would be multiple ways of defining the similarity, depending on the specific research purpose. 

(1) Treat 1 latent dimension as the functional dimension. Only compares the cos similarity along this dimension (but we need to first divide the dimension into several bins). Or we only allow one storage of the inventory, and calculate the similarity of the resources

(2) Treat more than 1 latent dimensions as the functional dimensions. Do an aggregate cos sim. Or do the aggregated similarity.

## *How resources are populated in the environment?*

Initialized resources would be randomly placed in the environment.

we will in default use 8:2 ratio of agent population in the environment