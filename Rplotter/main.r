# Scirpt to generate training error
dataframe <- read.csv("C:/Users/PC/Desktop/NN/log/RMSE.txt", sep="")
plot(dataframe[,1], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "blue", ylim = c(0,1))
lines(dataframe[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "red")
legend("topright", legend=c("RMSE", "ACC"), col=c("blue", "red"), lty=1:2, cex=0.8)

png(file="C:/Users/PC/Desktop/NN/Rplotter/MSE_ACC_600.png" ,width=1920, height=1080)
plot(dataframe[,1], type = 'l', xlab = "Epoch", ylab = "RMSE", col = "blue", ylim = c(0,1))
lines(dataframe[,2], type = 'l', xlab = "Epoch", ylab = "RMSE", col = "red")
legend("topright", legend=c("RMSE", "ACC"), col=c("blue", "red"), lty=1:2, cex=0.8)
dev.off()
