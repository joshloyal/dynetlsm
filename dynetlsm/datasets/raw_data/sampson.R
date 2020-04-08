library(ergm)

data(samplk)

Y1 <- as.matrix(samplk1)
print(colnames(Y1))
write.table(Y1,
            file=paste0('sampson_', 0, '.npy'),
            col.names=FALSE, row.names=FALSE)

Y2 <- as.matrix(samplk2)
write.table(Y2,
            file=paste0('sampson_', 1, '.npy'),
            col.names=FALSE, row.names=FALSE)

Y3 <- as.matrix(samplk3)
write.table(Y3,
            file=paste0('sampson_', 2, '.npy'),
            col.names=FALSE, row.names=FALSE)
