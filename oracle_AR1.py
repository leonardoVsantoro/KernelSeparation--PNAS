from utils import *

model = AR1_model(alpha = .5, eps = .25) (d = 50); 
# AR1 (decreasing correlation) model for two-sample testing:
        #  - X ~ N(0,\Sigma_1), where \Sigma_1(i,j) = alpha^|i-j|
        #  - Y ~ N(0,\Sigma_2), where \Sigma_2(i,j) = (alpha + eps)^|i-j|

kernel_type = 'euclidean' # Laplace kernel

Ns = [75, 125, 175,225] # sample sizes
K = 100 # number of MC repetitions

np.random.seed(0) # for reproducibility
data = MC_test_vals(model, Ns, K, kernel_type) # compute MC test values for KLR and MMD under null and alternative hypotheses
fig = res_plot(data) # plot results

fig.savefig("AR1.png", dpi=300, bbox_inches='tight')