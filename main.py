# %% Importing the necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler

# %% Importing the dataset
dataset = pd.read_csv("dataset/airline_passenger_satisfaction.csv", sep=",")
dataset.head()

# %% General information about the DataFrame
dataset.shape

# %% Analyzing descriptive statistics
dataset.describe()

# %% Checking if that there are no non-null values in the data 
dataset.info()

dataset.isnull().sum()

dataset.isna().sum()

# %% Removing null values from the dataset
dataset.dropna(inplace=True)

# %% Removing categorical columns from the dataset
# Since the goal of this project is to analyze the PCA method, we should keep only numerical columns.

# Keep only numerical columns (int or float)
dataset_pca = dataset.select_dtypes(include=['int64', 'float64'])
dataset_pca.head()  # Preview the first rows

# Remove the ID column as well
dataset_pca.drop(columns=["ID"], inplace=True)
dataset_pca.head()  # Preview again after dropping the ID column
dataset_pca.dtypes   # Check the data types of the remaining columns

# %% Analyzing correlation between variables
corr = dataset_pca.corr()

# Create the figure
plt.figure(figsize=(18,12), dpi=600) 

# Create the heatmap
sns.heatmap(
    corr,                  # Correlation matrix
    cmap='Blues',          # Color map (shades of blue)
    vmax=1,                # Maximum value for color scale
    vmin=-1,               # Minimum value for color scale
    center=0,              # Central value of the scale
    square=True,           # Square cells
    linewidths=.5,         # Width of lines separating cells
    annot=True,            # Write values inside the cells
    fmt='.2f',             # Format numbers (2 decimal places)
    annot_kws={'size':8},  # Font size of numbers inside cells
    cbar_kws={"shrink":0.50}  # Shrink the color bar
)

# Add title and adjust layout
plt.title('Correlation Matrix', fontsize=14)  # Graph title
plt.tight_layout()                             # Adjust layout so nothing is cut
plt.tick_params(labelsize=10)                  # Size of axis labels

# Show the plot
plt.show()

# %% Bartlett's Test of Sphericity
# Null hypothesis (H0): The correlation matrix is equal to the identity matrix
#                        → variables are NOT correlated
# Alternative hypothesis (H1): The correlation matrix is NOT the identity
#                               → variables are correlated

bartlett, p_value = calculate_bartlett_sphericity(dataset_pca)

print(f'Bartlett\'s Chi-square: {round(bartlett, 2)}')
print(f'p-value: {round(p_value, 4)}')

# %% Conclusion based on the p-value
if p_value < 0.05:
    print("Conclusion: p-value < 0.05 → reject H0 → variables are correlated. Applying PCA makes sense.")
else:
    print("Conclusion: p-value ≥ 0.05 → do not reject H0 → variables are NOT correlated. PCA may not be appropriate.")

# %% PCA Method - Standardizing the variables
# Create the scaler
scaler = StandardScaler()

# Fit and transform the numerical data
dataset_pca_scaled = scaler.fit_transform(dataset_pca)

# Convert back to DataFrame, keeping the column names
dataset_pca_scaled = pd.DataFrame(dataset_pca_scaled, columns=dataset_pca.columns)

# Display the first rows
dataset_pca_scaled.head()

# %% Defining PCA (initial procedure with all possible factors)
# Get the number of variables to start the procedure
n_columns = dataset_pca_scaled.shape[1]

fa = FactorAnalyzer(n_factors=n_columns, method='principal', rotation=None).fit(dataset_pca_scaled)

# %% Obtaining eigenvalues resulting from the FactorAnalyzer function
eigenvalues = fa.get_eigenvalues()[0]

print(eigenvalues)

# Sum of eigenvalues – equal to the number of variables
round(eigenvalues.sum(), 2)

#%% Eigenvalues, variances, and cumulative variances
autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Cumulative variance plot of the principal components
plt.figure(figsize=(18,12), dpi=600)
ax = sns.barplot(x=tabela_eigen.index, y='Variância', hue=tabela_eigen.index, palette='rocket', data=tabela_eigen)
for container in ax.containers:
    labels = [f"{v*100:.2f}%" for v in container.datavalues]
    ax.bar_label(container, labels=labels)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
plt.title("Extracted Factors", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} factors explaining {round(tabela_eigen['Variância'].sum()*100,2)}% of the variance", fontsize=8)
plt.ylabel("Explained Variance", fontsize=8)
plt.show()

#%% Determining factor loadings
factor_loadings = fa.loadings_

loadings_table = pd.DataFrame(factor_loadings)
loadings_table.columns = [f"Factor {i+1}" for i, v in enumerate(loadings_table.columns)]
loadings_table.index = dataset_pca_scaled.columns

print(loadings_table)

#%% Kaiser Criterion - automatically select factors with eigenvalue > 1
# Get the eigenvalues
eigenvalues = fa.get_eigenvalues()[0]

# Select only factors with eigenvalue > 1
kaiser_factors = np.sum(eigenvalues > 1)
print(f"Number of factors to retain according to the Kaiser Criterion: {kaiser_factors}")

# %% Re-running PCA with the selected parameters
fa_kaiser = FactorAnalyzer(n_factors=kaiser_factors, method='principal', rotation=None).fit(dataset_pca_scaled)

#%% Eigenvalues, variances, and cumulative variances for the 2 factors

# Note: values themselves don't change, only the selection of factors occurs

factor_eigenvalues = fa_kaiser.get_factor_variance()

eigen_table = pd.DataFrame(factor_eigenvalues)
eigen_table.columns = [f"Factor {i+1}" for i, v in enumerate(eigen_table.columns)]
eigen_table.index = ['Eigenvalue', 'Variance', 'Cumulative Variance']
eigen_table = eigen_table.T

print(eigen_table)

#%% Determining factor loadings
factor_loadings = fa_kaiser.loadings_

loadings_table = pd.DataFrame(factor_loadings)
loadings_table.columns = [f"Factor {i+1}" for i, v in enumerate(loadings_table.columns)]
loadings_table.index = dataset_pca_scaled.columns

print(loadings_table)

#%% Determining the communalities
communalities = fa_kaiser.get_communalities()

communalities_table = pd.DataFrame(communalities)
communalities_table.columns = ['Communalities']
communalities_table.index = dataset_pca_scaled.columns

print(communalities_table)

# Variables with higher communalities are well represented by the extracted factors,
# meaning the factors retain more information from these variables.

#%% Factor scores with factors selected using the Kaiser Criterion
# Use .transform() to get the scores for each observation
scores_kaiser = fa_kaiser.transform(dataset_pca_scaled)

# Convert to DataFrame
scores_table = pd.DataFrame(
    scores_kaiser,
    columns=[f"Factor {i+1}" for i in range(scores_kaiser.shape[1])],
    index=dataset_pca_scaled.index  # keep the observation indices
)

# View the first factor scores
print(scores_table.head())

# %% Conclusion
# After applying PCA, the 18 original variables were reduced to 5 factors,
# which explain 64.55% of the total variance, retaining most of the information from the dataset.

# %% End
