## Titanic Outcome Classification

## Description

The dataset I have chosen contains information about the people involved in the sinking of the Titanic. Despite the fact that it was a tragedy, I believe that by using artificial intelligence techniques, we can find a correlation between the survivors and unexpected characteristics such as the class in which they purchased their ticket or even the city from which they boarded the ship. These correlations can bring us one step closer to understanding why this accident occurred, what advantages the survivors had, and how we could prevent such a catastrophe in the future.

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
Each table contains 12 columns with information such as name, gender, ticket class, embarkation location, but the most important column is the one that indicates whether the passenger survived or not. The biggest issue with the dataset is that the data is incomplete. Some possible ways to address this issue include:
    + Completely removing columns from both the training and test sets if more than half of the values are missing. 
    + Replacing missing fields in a column with a predefined value or random values. 
To be able to use a Machine Learning model, I had to convert text or object-type values into numerical values. However, before doing this, I removed columns that I did not consider relevant for the classification process. I removed these columns from both the training and test sets. These columns are: 'Name', 'PassengerId', 'Ticket', and 'Cabin'. The reasoning behind this decision for each column is as follows:
    +   'Name': I believe that a person's name does not have a real impact on the model’s efficiency.
    +   'PassengerId': This represents the entry number in the table, and since there is no information on how people were added to the table, I assumed the order was random, making it an irrelevant feature for the model.
    +   'Ticket': This field consists of a random sequence of alphanumeric codes, and I did not observe any correlation between the ticket code and the class or cabin where the ticket was purchased.
    +   'Cabin': I removed this column because, in the initial dataset, nearly three-quarters of the values are missing, both in the training set (687/891) and the test set (327/418). For each column with missing data, I replaced the missing values with predefined random values.

- **Models Used**: KNeighborsClassifier / Random Forest / GaussianNB
- **Fine-tuning Hyperparameters**: 
The Scikit-learn implementation of the three classification algorithms includes multiple hyperparameters that influence the final model results on the test data. For example, for the KNeighborsClassifier algorithm, I used the following parameters: 'n_neighbors', 'leaf_size', 'weights', and 'algorithm' to find the best combination for the model. I set a different number of values for each hyperparameter. According to the code below, there are 288 possible hyperparameter combinations. To find the optimal set of parameters, we will use the Random Search technique. There is certainly at least one model that achieves the highest possible accuracy on the test set. In the random search process, different parameter combinations are randomly selected. With each choice we make, the likelihood of finding the best model increases. The confidence variable represents the degree of certainty that the best-performing combination of hyperparameters found so far is indeed the best possible one.

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

In conclusion, all three models are robust and yield good results, but the Random Forest model achieves significantly better accuracy, making it more suitable for my dataset. The model using KNeighborsClassifier correctly classifies only 65% of cases on the test data and 78% on the training data, which may indicate an underfitting problem. The algorithm is not complex enough to learn sufficient information from the training data but has a good generalization capacity.
