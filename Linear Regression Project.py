import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

### Importing the dataset ###
data = pd.read_csv("Dataset.csv")

print("Correlation:", "\n", data.corr(), "\n")  # Correlation Coefficient

### Converting dataset columns into separate arrays ###
Hours = pd.DataFrame(data['Hours'])
Scores = pd.DataFrame(data['Scores'])

### Creating Linear regression model of the given dataset ###
lm = linear_model.LinearRegression()
model = lm.fit(Hours, Scores)
print("Coefficient m:", model.coef_, "\n")
print("Intercept c:", model.intercept_, "\n")
print("Accuracy of the model:", model.score(Hours, Scores), "\n")

### Predicting new Scores for given Hours of study ###
Hours_new = []
n = int(input("Enter number of elements: "))
for i in range(0, n):
    num = float(input("Enter number of Hours: "))
    Hours_new.append(num)
Hours_new = pd.DataFrame(Hours_new, columns=['Hours_new'])
Scores_predicted = model.predict(Hours_new)
Scores_predicted = pd.DataFrame(Scores_predicted, columns=['Scores_predicted'])
df = pd.concat([Hours_new, Scores_predicted], axis=1)
print(df)

### Visualizing the Result ###
data.plot(kind='scatter', x='Hours', y='Scores')
# Plotting the Regression line
plt.plot(Hours, model.predict(Hours), color='red', linewidth=2)
# plotting the Predicted Scores
plt.scatter(Hours_new, Scores_predicted, color='green', marker='+', linewidths=3, s=200)
plt.legend(['Linear regression model', 'Dataset', 'Predicted Score(s)'])
plt.title('Output')
plt.show()
