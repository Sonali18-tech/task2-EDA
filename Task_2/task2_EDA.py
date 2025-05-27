import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load your cleaned dataset
df = pd.read_csv("cleaned_titanic.csv")

# Basic info
print("Dataset Shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe())

# Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare distribution
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Boxplot: Age vs Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

# Boxplot: Fare vs Pclass
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare vs Passenger Class")
plt.show()

# Correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot (select key columns)
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

# Optional: Interactive scatter plot (useful for presentation)
fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                 hover_data=['Pclass', 'Sex_male', 'Embarked_S'])
fig.update_layout(title="Age vs Fare by Survival (Interactive)")
fig.show()

print("EDA completed on cleaned dataset.")
