from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dữ liệu Iris
iris = load_iris()
X = iris.data

# Tạo mô hình K-means với K=3
kmeans = KMeans(n_clusters=3)

# Phân nhóm dữ liệu
kmeans.fit(X)

# Lấy tọa độ các điểm trung tâm của các nhóm
centroids = kmeans.cluster_centers_

# Lấy nhãn của từng điểm dữ liệu
labels = kmeans.labels_

# Vẽ biểu đồ phân nhóm
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()