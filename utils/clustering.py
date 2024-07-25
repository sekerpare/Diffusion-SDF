from sklearn import decomposition
from scipy.cluster.vq import kmeans, vq
import torch


def cluster(feats, num_clusters=8):
    clustered_results = []
    pca_results = []

    for i in range(feats.shape[0]):  # batch size
        tensor = feats[i]
        flat_tensor = tensor.view(feats.shape[1], -1).T.numpy()  # tensor hata verdi

        centroids, _ = kmeans(flat_tensor, num_clusters)
        cluster_labels, _ = vq(flat_tensor, centroids)

        pca = decomposition.PCA(n_components=3)
        pca_feat = pca.fit_transform(flat_tensor)
        pca_results.append(pca_feat)

        cluster_labels_reshaped = cluster_labels.reshape(64, 64, 64)
        clustered_results.append(cluster_labels_reshaped)

    clustered_results_tensor = torch.tensor(clustered_results)
    pca_results = torch.tensor(pca_results)
    return clustered_results_tensor, pca_results