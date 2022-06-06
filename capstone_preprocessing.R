load("/Users/doyun/Downloads/share_data.RData")
setwd("/Users/doyun/Downloads/")
library(changepoint)

fls = ls()[1:36]

i <- 1
file_3sigma_2.0 <- list()
file_com_2.0 <- list()
for (file_list in fls){
  
  data = get(file_list)$IBP
  data_na = ifelse(data < boxplot(data)$stats[1,] | data > boxplot(data)$stats[5,], NA, data)
  file_3sigma_2.0[[i]] <- data_na
  
  data_na = data.frame(na.fill(na.approx(data_na), "extend"))
  
  png(filename=paste0('Hannan_3/0.15/',file_list,".png"))
  
  data_na <- as.numeric(data_na$na.fill.na.approx.data_na....extend..)
  
  v_start_end <- round((length(data_na))*0.15)
  data1 = data_na[1:v_start_end]
  v_start.PELT = cpt.var(data1, method = "PELT", penalty = "Hannan-Quinn")
  
  start = round(length(data_na)*0.85)
  end = length(data_na)
  
  v_end.AMOC = cpt.var(data_na[start:end], method = "AMOC")
  
  start_cut = cpts(v_start.PELT)[length(cpts(v_start.PELT))]
  end_cut = cpts(v_end.AMOC)
  
  par(mfrow = c(2,1))
  
  plot(ts(data_na))
  abline(v=start_cut, col="deeppink")
  abline(v=round(start+end_cut), col="blue")
  
  plot(ts(data_na[start_cut:round(start+end_cut)]))
  
  file_com_2.0[[i]] <- data_na[start_cut:round(start+end_cut)]
  
  i <- i + 1
  
  dev.off() 
}

length(data_na)
