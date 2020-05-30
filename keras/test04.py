from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
# We load some test data
data = load_diabetes()

# Put it in a data frame for future reference -- or you work from your own dataframe
df = pd.DataFrame(data['data'])

# Init LinearRegression object / class
lm = LinearRegression()

# We only want variables / columns 0, 1, 3, 4, 5 (this could be any column slice / mask you want to use)
predictor_variables = [0,1,3,4,5]
X = df[predictor_variables]
y = data['target']

# Split our data by 70% training (for fitting) and 30% testing (for prediction)
X_train, x_test, Y_train, y_test = train_test_split(X, y, train_size=.7)

# Fit our model
model = lm.fit(X_train, Y_train)

# Check accuracy / variance
score_r2 = model.score(x_test, y_test)

# If you want predicted values from the test data
predictions = model.predict(x_test)

# To look at test data with independent variables (predictors)
# we'll setup a summary dataframe to capture the output of our model predictions for our test data
summary_df = pd.DataFrame(x_test)
summary_df['target'] = y_test
summary_df = summary_df.rename(columns={0:"A", 1:"B", 3:"C", 4:"D", 5:"E"})

summary_df['target'] = y_test

# We capture our predictions in into a new column from the model.predict method
summary_df['prediction'] = predictions

print (summary_df)

# Additional metrics
print("R2 Score:", score_r2)
print("Intercept: ", lm.intercept_)
print("(Variable, Coef):", zip(predictor_variables, lm.coef_))

summary_df.plot(kind="scatter", x="target", y="prediction")