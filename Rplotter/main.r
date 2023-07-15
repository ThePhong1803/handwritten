# Scirpt to generate training error
dataframe1 <- read.csv("C:/Users/PC/Desktop/NN/log/RMSE_ReLU.txt", sep="")
dataframe2 <- read.csv("C:/Users/PC/Desktop/NN/log/RMSE_Sigmoid.txt", sep="")
plot(dataframe1[,1], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "blue", ylim = c(0,1))
lines(dataframe1[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "red")
lines(dataframe2[,1], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "green")
lines(dataframe2[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "yellow")
legend("topright", legend=c("RMSE_ReLU", "ACC_ReLU", "RMSE_Sigmoid", "ACC_Sigmoid"), col=c("blue", "red", "green", "yellow"), lty=1:2, cex=0.8)

png(file="C:/Users/PC/Desktop/NN/Rplotter/MSE_ACC_600.png" ,width=1920, height=1080)
plot(dataframe1[,1], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "blue", ylim = c(0,1))
lines(dataframe1[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "red")
lines(dataframe2[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "green")
lines(dataframe2[,2], type = 'l', xlab = "Epoch", ylab = "RMSE/ACC", col = "yellow")
legend("topright", legend=c("RMSE_ReLU", "ACC_ReLU", "RMSE_Sigmoid", "ACC_Sigmoid"), col=c("blue", "red", "green", "yellow"), lty=1:2, cex=0.8)
dev.off()

# learning rate 0.00135
# out 0.00135 1200 1200