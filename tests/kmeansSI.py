import numpy as np
from numpy.random.mtrand import rand

from si4ul import kmeans_si

np.random.seed(0)
N = 4
dim = 2
X = np.random.normal(0, 1, (N, dim))
k = 2
obs_model = kmeans_si.kmeans(X, k)
print(obs_model.labels_)

max_iter = 1000
comp = [0, 1, 0]
random_seed = 1
pci_gene = kmeans_si.pci_gene(X, comp, k, obs_model, max_iter, random_seed)
hpci_p_value = pci_gene.p_value
naive_p_value = pci_gene.p_value

print(pci_gene.param_si.stat)
print("hpci-p value : %f"%hpci_p_value)
print("naive-p value%f"%naive_p_value)
print("clustering labes", obs_model.labels_)