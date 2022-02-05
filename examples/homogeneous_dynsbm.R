#!/usr/bin/env Rscript
library(reticulate)
library(dynsbm)
library(matrixStats)

# Set the path to the Python executable file
use_python("~/.virtualenv/stat/bin/python", required = T)
dynetlsm <- import("dynetlsm")
sklearn <- import("sklearn")

args <- commandArgs(trailingOnly = TRUE)
seed <- as.integer(args[[1]])

out_dir <- 'results_dynsbm'
if (!dir.exists(out_dir)) {
    dir.create(out_dir)
}

compute.icl <- function(dynsbm){
    T <- ncol(dynsbm$membership)
    Q <- nrow(dynsbm$trans)
    N <- nrow(dynsbm$membership)
    pen <- 0.5*Q*log(N*(N-1)*T/2) + 0.25*Q*(Q-1)*T*log(N*(N-1)/2) # binary case
    if ("sigma" %in% names(dynsbm)) pen <- 2*pen # continuous case
    return(dynsbm$loglikelihood - ifelse(T>1,0.5*Q*(Q-1)*log(N*(T-1)),0) - pen)
}

res <- dynetlsm$datasets$synthetic_static_community_dynamic_network(
    n_time_steps=6L, n_nodes=120L, random_state=seed,
    sigma_scale=2, sigma_shape=10, sticky_const = 20,
    simulation_type = 'hard',
    intercept=1.0)
Y <- res[[1]]
z <- res[[3]]
sim_res <- list()

models <- select.dynsbm(Y, Qmin=1, Qmax=8, edge.type="binary",
                        nstart=10, fixed.param = FALSE, nb.cores = 8)

# num of clusters maximizing ICL
icls <- sapply(models, compute.icl)
sim_res['num_clusters'] <- which.max(icls)

# estimates based on the true number of clusters (G = 6)
sbm <- models[[6]]
sim_res['rand_index'] <- sklearn$metrics$adjusted_rand_score(
    as.vector(t(z)), as.vector(sbm$membership))
sim_res['vi'] <- dynetlsm$metrics$variation_of_information(
    as.vector(t(z)), as.vector(sbm$membership))

# average statistics
avg_rand <- 0
avg_vi <- 0
for (t in 1:dim(Y)[1]) {
    avg_rand <- avg_rand + sklearn$metrics$adjusted_rand_score(z[t,], sbm$membership[,t])
    avg_vi <- avg_vi + dynetlsm$metrics$variation_of_information(z[t,], sbm$membership[,t])
}
sim_res['avg_rand'] <- avg_rand / dim(Y)[1]
sim_res['avg_vi'] <- avg_vi / dim(Y)[1]

# in-sample AUC
probas <- array(0, dim=dim(Y))
for (t in 1:dim(Y)[1]) {
    b <- sbm$beta[t,,]
    m <- sbm$membership[,t]
    Z <- array(0, dim=c(length(m), 6))
    for (i in 1:length(m)) {
        if (m[i] == 0) {
            Z[i,] <- diag(6)[1,]
        }
        else{
            Z[i,] <- diag(6)[m[i],]
        }
    }
    probas[t,,] <- Z %*% b %*% t(Z)
}
sim_res['insample_auc'] <- dynetlsm$metrics$network_auc(Y, probas)

# one-step ahead predictions
#b <- sbm$beta[dim(Y)[1],,]
#pi <- sbm$trans
#m <- sbm$membership[,6]

#n_nodes <- dim(Y)[2]
#n_dyads <- 0.5 * n_nodes * (n_nodes - 1)
#probas_ahead <- vector(mode='numeric', length=n_dyads)
#iter <- 0
#for (i in 1:n_nodes) {
#    for (j in 1:(i-1)) {
#        logp <- 0
#        for (q in 1:6) {
#            for (p in 1:6) {
#               logp <- logp + log(pi[m[i], q]) + log(pi[m[j], p]) + log(b[q, p])
#            }
#        }
#        probas_ahead[iter] <- exp(logSumExp(logp))
#        iter <-iter + 1
#    }
#}

df <- as.data.frame(sim_res)
file_name <- file.path(out_dir, paste0('benchmark_', seed, '.csv'))
write.csv(df, file_name, row.names = FALSE)