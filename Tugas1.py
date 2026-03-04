import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Baca dataset Mall Customers
# ==============================================================================
print("1. Membaca Dataset Mall Customers...")
df = pd.read_csv('https://raw.githubusercontent.com/vjchoudhary7/customer-segmentation-tutorial-in-python/master/Mall_Customers.csv')

# Tampilkan 5 baris pertama
print("\n5 baris pertama dari dataset:")
print(df.head())
print("-" * 50)

# ==============================================================================
# 2. Pilih fitur yang relevan (Annual Income dan Spending Score)
# ==============================================================================
print("2. Memilih fitur 'Annual Income (k$)' dan 'Spending Score (1-100)'...")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nFitur yang dipilih:")
print(X.head())
print("-" * 50)

# ==============================================================================
# 3. Standarisasi data
# ==============================================================================
print("3. Melakukan standarisasi data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nData setelah distandarisasi (5 baris pertama):")
print(X_scaled[:5])
print("-" * 50)

# ==============================================================================
# 4. Tentukan jumlah cluster optimal dengan Elbow Method
# ==============================================================================
print("4. Menentukan jumlah cluster optimal dengan Elbow Method...")
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.grid(True)
plt.show()
print("\nDari grafik, jumlah cluster optimal adalah 5.")
print("-" * 50)

# ==============================================================================
# 5. Terapkan K-Means Clustering dengan 5 cluster
# ==============================================================================
print("5. Menerapkan K-Means Clustering...")
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print("\nData dengan label cluster (5 baris pertama):")
print(df.head())
print("-" * 50)

# ==============================================================================
# 6. Visualisasi hasil clustering
# ==============================================================================
print("6. Memvisualisasikan hasil clustering...")
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', data=df, palette='viridis', s=100)

# Menambahkan centroid
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='*', label='Centroids')

plt.title('Segmentasi Pelanggan Menggunakan K-Means Clustering')
plt.xlabel('Pendapatan Tahunan (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 7. Interpretasi dan penamaan segmen
# ==============================================================================
print("7. Interpretasi Karakteristik Tiap Segmen:")
cluster_info = df.groupby('cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
print(cluster_info)

print("\n--- Interpretasi dan Penamaan Segmen ---")
print("Cluster 0: Pendapatan rendah, Spending Score rendah -> Pelanggan Hemat")
print("Cluster 1: Pendapatan tinggi, Spending Score rendah -> Pelanggan Konservatif")
print("Cluster 2: Pendapatan tinggi, Spending Score tinggi -> Pelanggan Target / Premium")
print("Cluster 3: Pendapatan rendah, Spending Score tinggi -> Pelanggan Boros")
print("Cluster 4: Pendapatan sedang, Spending Score sedang -> Pelanggan Rata-Rata")
print("-" * 50)

# ==============================================================================
# 8. Rekomendasi strategi pemasaran
# ==============================================================================
print("8. Rekomendasi Strategi Pemasaran Berdasarkan Hasil Clustering:")
print("a. Pelanggan Hemat: Tawarkan diskon besar dan promosi produk terjangkau.")
print("b. Pelanggan Konservatif: Tawarkan produk eksklusif dan layanan premium untuk mendorong pengeluaran.")
print("c. Pelanggan Target / Premium: Pertahankan dengan program loyalitas VIP, produk baru, dan layanan pribadi.")
print("d. Pelanggan Boros: Berikan promosi dan kupon yang menarik untuk menjaga frekuensi belanja.")
print("e. Pelanggan Rata-Rata: Lakukan kampanye yang fokus pada peningkatan engagement dan cross-selling.")