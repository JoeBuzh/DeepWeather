# DeepWeather
A Interesting Rain Forecast Model for Nowcasting.

### The shouder of Gaint

+ The framework of this DL is develped from Dr.Shi' research on 2015 NIPS.
+ Here is his website [sxjscience](https://sxjscience.github.io/)

### Features

+ END2END mode
+ Without professional datas like gfs, wrfout...
+ GPU avalible, tested in **Nvidia Quadro P4000**.

### Declear

+ This work was applied on some institutions, like airport climate center.
+ The work is not avalible for bussiness project due to is not the product version.
+ This work was a collabration of some engineers.
+ **The raw data of this projects was collected by another spider projects.**

### Structure

#### ./src

+ ./src/sparnn: forked from Dr.Shi
+ ./src/runtime: main codes contain training and prdicting 
+ ./src/tools: scripts
+ ./src/model: save training models

#### ./model
+ Models prepare for forecast

#### ./basemap
+ Base map with lines
+ Post preocess for adding geo-informations

#### ./presentation
+ pictures about test results
+ App implemention

### Futuer Work
+ Moving to higher framework liek Pytorch or Keras.
+ Applying more feature engineering and pre_processing.
+ Constructing a fullstack data project, consist of data crawl, prepocess, DL forecast, visualize in app. 
