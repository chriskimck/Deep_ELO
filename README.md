# Deep_ELO
Code for ELO: Deep Learning Final Project (Camille Chow and Christopher Kim)

The Datasets folder contains three csv files used in this project.
These datasets are the Kaggle* training and testing datasets, and the 2018 chess world championship dataset.
The format of these datasets are geared towards the way in which our model processes the data.
The original datasets can be found in the kaggle competition website.

The Data proprocessing folder contains two python scripts dealing with the data preprocessing.
chess_dataprocessing.py handled the original kaggle datasets. 
worlds_dataprocessing.py ran a stockfish 10 engine on the 15 played games in the 2018 chess world championship
and outputs a csv that we can use to test our model.

The Model folder contains five of the many dense and CNN models we've experimented with to perform well on the 
kaggle contest. The top model is our CNN_193.py model, which scored 193 on the public and private leaderboard, which
would put us at top 50 for the contest. Of course, there is so much room for improvement, but we are happy with 
what we accomplished simply by using stockfish evaluations (https://en.wikipedia.org/wiki/Stockfish_(chess)). 

*https://www.kaggle.com/c/finding-elo