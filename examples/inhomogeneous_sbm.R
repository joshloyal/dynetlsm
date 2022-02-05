library(reticulate)
library(blockmodels)
library(label.switching)
library(zeallot)

# Set the path to the Python executable file
use_python("~/.virtualenv/stat/bin/python", required = T)

#source_python('~/myworkspace/dynetlsm/examples/test.py')

dynetlsm <- import("dynetlsm")
sklearn <- import("sklearn")


sim_type <- 'easy'
out_dir <- paste0('inhomo_results_sbm_', sim_type)
if (!dir.exists(out_dir)) {
    dir.create(out_dir)
}


for (seed in 0:49) {
    res <- dynetlsm$datasets$synthetic_dynamic_network(
        n_time_steps=9L, n_nodes=120L, random_state=as.integer(seed),
        simulation_type = sim_type)
    Y <- res[[1]]
    Y <- Y[1:9,,]
    z <- res[[3]]
    z <- z[1:9,]
    sim_res <- list()

    # blockmodels
    n_time_steps <- dim(Y)[[1]]
    n_nodes <- dim(Y)[[2]]
    z_sbm <- matrix(0, nrow=n_time_steps, ncol=n_nodes)
    p_sbm <- array(0, dim = c(n_time_steps, n_nodes, 6))
    probas <- array(0, dim=dim(Y))
    avg_rand <- 0
    avg_vi <- 0
    n_clusters <- c(2, 2, 2, 6, 6, 6, 4, 4, 4)
    for (t in 1:n_time_steps) {
        sbm_models <- BM_bernoulli('SBM_sym', Y[t,,], explore_min=8,
                                   exploration_factor=1., ncores=8, verbosity=0)
        sbm_models$estimate()

        p_sbm[t,,1:n_clusters[t]] <- sbm_models$memberships[[n_clusters[t]]]$Z
        z_sbm[t,] <- apply(p_sbm[t,,], 1, which.max)
        sim_res[paste0('num_clusters_', t)] <- which.max(sbm_models$ICL)
        avg_rand <- avg_rand + sklearn$metrics$adjusted_rand_score(as.vector(z[t,]), z_sbm[t,])
        avg_vi <- avg_vi + dynetlsm$metrics$variation_of_information(as.vector(z[t,]), z_sbm[t,])

        b <- sbm_models$model_parameters[[n_clusters[t]]]$pi
        m <- z_sbm[t,]
        Z <- array(0, dim=c(length(m), n_clusters[t]))
        for (i in 1:length(m)) {
            if (m[i] == 0) {
                Z[i,] <- diag(n_clusters[t])[1,]
            }
            else{
                Z[i,] <- diag(n_clusters[t])[m[i],]
            }
        }
        probas[t,,] <- Z %*% b %*% t(Z)
    }
    sim_res['avg_rand'] <- avg_rand / dim(Y)[1]
    sim_res['avg_vi'] <- avg_vi / dim(Y)[1]
    sim_res['insample_auc'] <- dynetlsm$metrics$network_auc(Y, probas)

    res <- label.switching("ECR", zpivot=z_sbm[1,], z=z_sbm, K=6)
    perm <- res$permutations$ECR
    for (t in 1:n_time_steps) {
        p_sbm[t,,] <- p_sbm[t,,perm[t,]]
        z_sbm[t,] <- apply(p_sbm[t,,], 1, which.max)
    }
    sim_res['rand_index'] <- sklearn$metrics$adjusted_rand_score(as.vector(t(z)), as.vector(t(z_sbm)))
    sim_res['vi'] <- dynetlsm$metrics$variation_of_information(as.vector(t(z)), as.vector(t(z_sbm)))

    df <- as.data.frame(sim_res)
    file_name <- file.path(out_dir, paste0('benchmark_', seed, '.csv'))
    write.csv(df, file_name, row.names = FALSE)
}
