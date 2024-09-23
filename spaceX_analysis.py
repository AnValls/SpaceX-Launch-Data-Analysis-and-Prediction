import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

csv_path = 'C:\\Users\\Ana Valls\\Desktop\\CURSO BIG DATA\\IBM\\PROJECTS\\SpaceX\\space_y_launch_data.csv'
data = pd.read_csv(csv_path)

print(data.head())

print(data.info())

data = data.dropna()

data = pd.get_dummies(data, columns=['launch_site', 'booster_version'])

print(data.head())

sns.scatterplot(x='payload_mass', y='outcome', data=data)
plt.show()

corr_matrix = data.drop(columns=['launch_date']).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

X = data[['payload_mass', 'reused', 'booster_version_Falcon 9 Block 5', 'booster_version_Falcon Heavy', 
          'launch_site_CCAFS LC-40', 'launch_site_KSC LC-39A', 'launch_site_VAFB SLC-4E']]  # Características
y = data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cost_model = LinearRegression()
cost_model.fit(X_train, y_train)

y_pred_cost = cost_model.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_cost))
