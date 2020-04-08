library(igraph)
library(tidygraph)
library(tidyverse)

data <- read_csv('alliance_v4.1_by_dyad_yearly.csv') %>%
    rename(from = state_name1, to = state_name2) %>%
    select(from, to, year, defense)

names <- as_tbl_graph(data) %>%
    activate(nodes) %>%
    as_tibble()

write_csv(names, 'names.csv')

step_size <- 5
for (year_id in seq(1950, 1975, by = step_size)) {
    graph <- as_tbl_graph(data) %>%
        activate(edges) %>%
        filter(year >= year_id) %>%
        filter(year < (year_id + step_size))

    Y <- as_adjacency_matrix(graph, sparse = FALSE)
    write.table(Y, file=paste0('network_', year_id, '.npy'),
                col.names = FALSE, row.names = FALSE)
}
