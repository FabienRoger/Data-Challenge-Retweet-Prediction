# Data Challenge: Retweet Prediction

_Fabien Roger, Benjamin Hebras, Dannel Cassuto_

This is a submission to a data challenge aiming at predicting the number of retweets of French tweets.

## How to Run

1. Drop `train.csv` & `evaluation.csv` in the `code/data` folder;
2. Go to the code folder (run `cd code`);
3. Run `pip install -r requirements.txt`, make sure you are using Python 3.9 or above;
4. To generate our best submission, run `python main_predictions.py`, it will take approximately 20 minutes;
5. To regenerate our experimental results, run `python experiments.py`, it will take over two hours, and will generate the data in `code/experiments_data`, (you can disable some of them by using the dictionary at the start of the file);
6. To regenerate our figures, run `python figures.py`, they will be saved in `code/figures`.
