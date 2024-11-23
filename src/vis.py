import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Load the dataset
file_path = 'method_clusters.csv'  # Adjust path as needed
df = pd.read_csv(file_path)

# Select the features to include in PCA (excluding 'name' and 'cluster')
features = ['fan_in', 'fan_out', 'loc', 'cyclomatic_complexity', 'num_parameters', 'pca1', 'pca2']
X = df[features]

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Add PCA results back to the DataFrame
df['pca_dim1'] = X_pca[:, 0]
df['pca_dim2'] = X_pca[:, 1]
df['pca_dim3'] = X_pca[:, 2]
df['pca_dim4'] = X_pca[:, 3]  # 4th dimension for color encoding

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define update function for animation
def update(num):
    ax.clear()
    scatter = ax.scatter(df['pca_dim1'], df['pca_dim2'], df['pca_dim3'], 
                         c=df['cluster'], cmap='viridis', alpha=0.7)
    ax.set_title(f'4D Visualization (Frame {num})')
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_zlabel('PCA Dimension 3')
    return scatter,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=200)
plt.show()

