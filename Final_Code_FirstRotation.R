#Libraries

library(readxl)
library(data.table)
library(tidyverse)
library(dplyr)
library(devtools)
library(ggcorrplot)
library(car)
library(ggpubr)
library(glmnet)
library(summarytools)
library(knitr)
library(htmltools)
library(corrplot)
library(caret)
library(factoextra)
library(Metrics)
library(readr)
library(gplots)
library(dplyr)
library(stringr)
library(readxl)
library(plotly)
library(e1071)
library(ggplot2)
library(reshape2)
library(multtest)
library(ROCR)
library(gridExtra)




#LASSO ####
Hazeldine_1h <- read_csv("CF_ISS1.csv")
savee<-as.factor(Hazeldine_1h$Label)
names(Hazeldine_1h)<-sapply(1:dim(Hazeldine_1h)[2], function(i){paste0("First ",names(Hazeldine_1h)[i])})
Hazeldine_1h<-Hazeldine_1h[,-(1:3)]
Hazeldine_1h<-Hazeldine_1h[,-(77:78)]

Hazeldine_2h <- read_csv("CF_ISS2.csv")
names(Hazeldine_2h)<-sapply(1:dim(Hazeldine_2h)[2], function(i){paste0("Second ",names(Hazeldine_2h)[i])})
Hazeldine_2h<-Hazeldine_2h[,-(1:3)]
Hazeldine_2h<-Hazeldine_2h[,-(79:80)]

Hazeldine_3h <- read_csv("CF_ISS3.csv")
Hazeldine_3h<-Hazeldine_3h[,-(1:3)]
names(Hazeldine_3h)<-sapply(1:dim(Hazeldine_3h)[2], function(i){paste0("Third ",names(Hazeldine_3h)[i])})


All2<-data.frame(Hazeldine_1h,Hazeldine_2h,Hazeldine_3h,Label=savee)

#Tienen todos el mismo ID? SI!

#TRAIN

n<-50 #esta en 35/40 el filtro y coge 9 features
N<-10
nn<-25

ErrorsFinEN<-vector(mode="double", length=n)
BetasFinEN<-vector(mode="character", length=n)
LambdaFinEN<-vector(mode="double", length=n)
BNumFinEN<-vector(mode="double", length=n)
see2EN<-data.frame(All="All")
LauCoef1<-data.frame(Coeff="See",stringsAsFactors=FALSE)
BetasTodo<-data.frame(Features="Name",Coefficients=1)

ListError<-vector(mode="double", length=n)
BetasFin<-vector(mode="character", length=n)
LambdaFin<-vector(mode="double", length=n)
BNumFin<-vector(mode="double", length=n)
see2<-data.frame(All="All")
LauCoef1L<-data.frame(Coeff="See",stringsAsFactors=FALSE)
BetasTodoL<-data.frame(Features="Name",Coefficients=1)

for (i in 1:n){
  
  smp_size = floor(0.75 * nrow(All2))
  #set.seed(907)- want it to be random in every 75 loops.
  train_ind = sample(seq_len(nrow(All2)), size = smp_size)
  
  #Training set
  train = All2[train_ind, ]
  
  #Test set
  test = All2[-train_ind, ]
  
  #Creates matrices for independent and dependent variables.
  xtrain = model.matrix(Label~. -1, data = train)
  ytrain = train$Label
  xtest = model.matrix(Label~. -1, data = test)
  ytest = test$Label
  
  
  #Choose lambda value that minimize missclassification error. 
  #0.5 as elastic nets, all variables with EN are based on ElasticNets analysis. 100 lambdas sampled with 10 cross validation for each, already internalized in method
  CVEN=cv.glmnet(xtrain,ytrain,family="binomial",type.measure="class",alpha=0.5,nlambda=100)
  attach(CVEN)
  Lambda.BestEN<-CVEN$lambda.min #can be either minimum or 1 standard deviation
  print(Lambda.BestEN)
  
  CVFinEN=glmnet(xtrain,ytrain,family="binomial",alpha=0.5,lambda=Lambda.BestEN) 
  CoefEN<-coef(CVFinEN) #Beta coefficients obtained from here
  InterceptEN<-CoefEN@x[1]
  BetasEN<-CVFinEN$beta 
  Betas2EN<-data.frame(Features=BetasEN@Dimnames[[1]][BetasEN@i+1], Coefficients=BetasEN@x) #Beta coefficients names stored here 
  CVPred1EN = predict(CVFinEN, family="binomial", s=Lambda.BestEN, newx = xtest,type="class") #predict in test set to obtain confusion matrix
  
  #Calculate error for categorical values
  ytest2<-as.factor(ytest)
  ResultsEN<-table(CVPred1EN,ytest)
  confusionMatrix(CVPred1EN,ytest)
  AccuracyEN<-(ResultsEN[1]+ResultsEN[4])/sum(ResultsEN[1:4])
  ErrorEN<-1-AccuracyEN
  
  LauCoef<-Betas2EN$Coefficients
  LauCoefEN<-data.frame(Coeff=LauCoef,stringsAsFactors=FALSE)
  LauCoef1<-rbind(LauCoef1,LauCoefEN)
  BetasTodo<-rbind(BetasTodo,Betas2EN) #store coefficients and store betas
  
  seeEN<-Betas2EN$Features 
  seeEN1<-data.frame(All=seeEN)
  see2EN<-rbind(see2EN,seeEN1)   #all beta names stored
  
  
  mEN<-count(see2EN, All) #frequency of the betas stored counted
  see3EN<-toString(seeEN) 
  ErrorsFinEN[i]<-ErrorEN #error of the model stored
  BetasFinEN[i]<-see3EN #name of features the model used 
  BNumFinEN[i]<-length(seeEN) #number of features the model used 
  LambdaFinEN[i]<-Lambda.BestEN #lambda chosen for model
  detach(CVEN)
  
  #Change between Lasso and EN, alpha=1 (*)
  CV=cv.glmnet(xtrain,ytrain,family="binomial",type.measure="class",alpha=1,nlambda=100) 
  
  attach(CV)
  
  Lambda.Best<-CV$lambda.min
  CVFin=glmnet(xtrain,ytrain,family="binomial",alpha=1,lambda=Lambda.Best)
  Coef<-coef(CVFin)
  Intercept<-Coef@x[1]
  Betas<-CVFin$beta
  Betas2<-data.frame(Features=Betas@Dimnames[[1]][Betas@i+1], Coefficients=Betas@x)
  CVPred1 = predict(CVFin, family="binomial", s=Lambda.Best, newx = xtest,type="class")
  library(MLmetrics)
  
  
  #Calculate error for categorical values
  ytest2<-as.factor(ytest)
  confusionMatrix(CVPred1,ytest)
  Results<-table(CVPred1,ytest)
  Accuracy<-(Results[1]+Results[4])/sum(Results[1:4])
  Error<-1-Accuracy
  
  #visual display of for
  
  BetasTodoL<-rbind(BetasTodoL,Betas2)
  see<-Betas2$Features
  see1<-data.frame(All=see)
  see2<-rbind(see2,see1)
  m<-count(see2, All)
  
  see3<-toString(see)
  ListError[i]<-Error
  BetasFin[i]<-see3
  BNumFin[i]<-length(see)
  LambdaFin[i]<-Lambda.Best
  detach(CV)
  
}

#Visualizing data from LASSO and EN ####


#obtain in a data frame all error, betas names, number and lamda for the 75 models for each lasso and EN
All_info<-data.frame(Error=ListError, BetasNames=BetasFin, BetasNum=BNumFin, Lambda=LambdaFin) 
All_infoEN<-data.frame(Error=ErrorsFinEN, BetasNames=BetasFinEN, BetasNum=BNumFinEN, Lambda=LambdaFinEN)

m<-m[-1,]
mEN<-mEN[-1,]

Final_LASSO<-m[order(-m$n),] #order highest frequencies above and filter with those that appear more than 60 times 
Final_LASSO1<-filter(Final_LASSO,n>20)

outputVenn2<-venn(list(EN= Final_EN$All, LASSO = Final_LASSO$All))

Final_EN<-mEN[order(-mEN$n),]
Final_EN1<-filter(Final_EN,n>20)
Final_Plot_Names<-filter(Final_EN,n>20)

outputVenn<-venn(list(EN= Final_EN1$All, LASSO = Final_LASSO1$All))

Freqs<-m[order(-m$n),]
num<-length(Freqs$All)

Freqs$All <- factor(Freqs$All, levels = Freqs$All[order(-Freqs$n)]) #plot in a bar graph the frequencies of ocurrance of the betas and order from highest to smalles
ggplot(Freqs, aes(All, n))+geom_bar(stat="identity")+theme(axis.text.x = element_text(size=8, angle=90))+ggtitle("LASSO features")

FreqsEN<-mEN[order(-mEN$n),]
numEN<-length(FreqsEN$All)

FreqsEN$All <- factor(FreqsEN$All, levels = FreqsEN$All[order(-FreqsEN$n)])
ggplot(FreqsEN, aes(All, n))+geom_bar(stat="identity")+theme(axis.text.x = element_text(size=8, angle=90))+ggtitle("EN features")

#Boxplot with Betas and its coefficients

Boxplot1<-BetasTodo[BetasTodo$Features %in% Final_EN1$All,] #see which features appear in the filtered features and obtain the coefficients associated. 
ggplot(Boxplot1,aes(Boxplot1$Features,Boxplot1$Coefficients))+geom_boxplot()+geom_jitter()
Boxplot1["Method"]<-as.factor("EN")

Boxplot2<-BetasTodoL[BetasTodoL$Features %in% Final_LASSO1$All,]
ggplot(Boxplot2,aes(Boxplot2$Features,Boxplot2$Coefficients))+geom_boxplot()+geom_jitter()
Boxplot2["Method"]<-as.factor("LASSO")

Fin_Boxplot<-rbind(Boxplot1,Boxplot2) #Unite both boxplots LASSO and EN 
ggplot(Fin_Boxplot,aes(Fin_Boxplot$Features,Fin_Boxplot$Coefficients))+geom_boxplot(aes(color=Method))+geom_jitter()+theme(axis.text.x = element_text(size=12, angle=90))+ggtitle("Beta coefficients EN and LASSO")+ labs(x = "Features",y="Beta Coefficients")

All_Feat<-rbind(Final_LASSO1,Final_EN1)
All_Feat2<-unique(All_Feat$All)#select the filtered betas in both. 
#All_Feat2<-data.frame(All_Feat2)
Betas_select<-All2[,colnames(All2[,intersect(gsub("`", "", All_Feat2), colnames(All2))])]
Betas_select["Label"]<-All2$Label
Betas_select<-data.frame(Betas_select)

NumVar<-length(Betas_select)
maxCombF<-data.frame(Name=toString("Trial"),AUC=0)
auc3max<-0
cont<-0
cont2<-0


for (p in 1:N){ #1000 different measurements of AUC values, mean done at the end. 
  
  smp_size<-floor(0.65 * nrow(Betas_select))
  #set.seed(907)
  train_ind<-sample(seq_len(nrow(Betas_select)), size = smp_size)
  
  # Training set
  train<-Betas_select[train_ind, ]
  
  # Test set
  test<-Betas_select[-train_ind, ]
  
  xtrain<-model.matrix(Label~. -1, data = train)
  ytrain<-train$Label
  xtest<-model.matrix(Label~. -1, data = test)
  ytest<-test$Label
  
  xtest<-data.frame(xtest)
  xtrain<-data.frame(xtrain)
  
  y<-Betas_select[,NumVar]
  X<-Betas_select[,1:(NumVar-1)]
  
  
  levels(ytrain)[1]<-"0"
  levels(ytrain)[2]<-"1"
  levels(ytest)[1]<-"0"
  levels(ytest)[2]<-"1"
  
  
  for (k in 1:nn){
    columns<-c(1:dim(xtrain)[2])
    columns<-sample(columns)
    d<-xtrain[,columns]
    for (i in 1:dim(xtrain)[2]){
      for (j in 1:dim(xtrain)[2]){
        yy3<-data.frame(Outcome=ytrain,d[i:j])
        model3<-glm(Outcome~.,data=yy3, family=binomial(link='logit'))
        dd<-xtest[,columns]
        new3<-data.frame(dd[i:j])
        p3<-predict(model3,new3,type="response")
        new5<-data.frame(Outcome=ytest)
        pr3 <- prediction(p3, new5)
        prf3 <- performance(pr3, measure = "tpr", x.measure = "fpr")
        auc3 <- performance(pr3, measure = "auc")
        auc3 <- auc3@y.values[[1]]
        result2<-auc3
        if (auc3>auc3max){
          if (i>j){
            maxComb<-data.frame(Name=toString(names(d)[j:i]),AUC=auc3)
            auc3max<-auc3
            cont<-cont+1
          }
          else{
            
            maxComb<-data.frame(Name=toString(names(d)[i:j]),AUC=auc3)
            auc3max<-auc3
            cont<-cont+1
          }
        }
        else{
          cont2<-cont2+1
        }
        
      }
    }
  }
  maxCombF<-rbind(maxCombF,maxComb)
  auc3max<-0
}

maxCombF2<-maxCombF[order(maxCombF$AUC),]
names<-maxCombF2$Name[dim(maxCombF2)[1]]
names1<-as.character(names)
names1<-strsplit(names1,", ")
names<-as.data.frame(names1)

Betas_select2<-All2[,colnames(All2[,intersect(gsub("`", "", names[,1]), colnames(All2))])]
Betas_select2["Label"]<-savee
#Get AUC and ROC curves #####
Betas_select2<-as.data.frame(Betas_select2)

NumVar<-length(Betas_select2)
ISSs<-c("Third.NISS","Third.ISS")

where<-match(colnames(Betas_select2),ISSs)
for (u in 1:dim(Betas_select2)[2]){
  if (!(is.na(where[u]))){
    if (where[u]==1){
      CompareISS<-u
      NISShere<-1
    }else{
      CompareISS<-u
      NISShere<-0
    }
  }
}

N<-10

multipleAUC<-matrix(rnorm(2),1,N) 
multipleAUCR<-matrix(rnorm(2),1,N) 
multipleAUCNB<-matrix(rnorm(2),1,N) 
multipleAUCNBR<-matrix(rnorm(2),1,N) 

multipleROC<-matrix(as.list(rnorm(2)),1,N)  
multipleROCR<-matrix(as.list(rnorm(2)),1,N)  
multipleNBROC<-matrix(as.list(rnorm(2)),1,N) 
multipleNBROCR<-matrix(as.list(rnorm(2)),1,N) 

singleROC<-list()
doubleROC<-list()
singleROCR<-list()
doubleROCR<-list()

doublePlus<-list()
singlePlus<-list()

singleAUC<-matrix(rnorm(2),NumVar-1,N)    
doubleAUC<-matrix(rnorm(2),(NumVar-1),N) 
singleAUCR<-matrix(rnorm(2),NumVar-1,N)    
doubleAUCR<-matrix(rnorm(2),(NumVar-1),N)

doubleAUCSVMR<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCSVM<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCSVMCrossR<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCSVMCross<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCRFCross<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCRFCrossR<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCNBR<-matrix(rnorm(2),(NumVar-1),N) 
doubleAUCNB<-matrix(rnorm(2),(NumVar-1),N) 

MatsingleROC<-matrix(as.list(rnorm(2)),NumVar-1,N)  
MatsingleROCR<-matrix(as.list(rnorm(2)),NumVar-1,N)  
MatdoubleROC<-matrix(as.list(rnorm(2)),(NumVar-1),N) 
MatdoubleROCR<-matrix(as.list(rnorm(2)),(NumVar-1),N) 

MatsinglePlus<-matrix(as.list(rnorm(2)),NumVar-1,N) 
MatdoublePlus<-matrix(as.list(rnorm(2)),(NumVar-1),N)
MatsinglePlusR<-matrix(as.list(rnorm(2)),NumVar-1,N) 
MatdoublePlusR<-matrix(as.list(rnorm(2)),(NumVar-1),N)


for (j in 1:N){ #1000 different measurements of AUC values, mean done at the end. 
  
  smp_size<-floor(0.65 * nrow(Betas_select))
  #set.seed(907)
  train_ind<-sample(seq_len(nrow(Betas_select)), size = smp_size)
  
  # Training set
  train<-Betas_select[train_ind, ]
  
  # Test set
  test<-Betas_select[-train_ind, ]
  
  xtrain<-model.matrix(Label~. -1, data = train)
  ytrain<-train$Label
  xtest<-model.matrix(Label~. -1, data = test)
  ytest<-test$Label
  
  xtest<-data.frame(xtest)
  xtrain<-data.frame(xtrain)
  
  y<-Betas_select[,NumVar]
  X<-Betas_select[,1:(NumVar-1)]
  
  
  levels(ytrain)[1]<-"0"
  levels(ytrain)[2]<-"1"
  levels(ytest)[1]<-"0"
  levels(ytest)[2]<-"1"
  
  library(ggplot2)
  library(reshape2)
  library(multtest)
  library(ROCR)
  
  
  source("/Users/lauri/Desktop/R/common_pipeline/AUCFun29-3.R")
  
  
  multipleAUCNB[1,j]<-multipleAUCfunNB(xtrain, ytrain,xtest,ytest)
  multipleAUC[1,j]<-multipleAUCfun(xtrain, ytrain,xtest,ytest)
  
  multipleNBROC[[j]]<-multipleROCfunNB(xtrain, ytrain,xtest,ytest)
  multipleROC[[j]]<-multipleROCfun(xtrain, ytrain,xtest,ytest)
  
  
  #separate
  
  for (s in (1:(NumVar-1))){
    s<-as.numeric(s)
    singleAUC[s,j]<-singleAUCfun(xtrain, ytrain,xtest,ytest,s) 
    singleROC[[s]]<-singleROCfun(xtrain, ytrain,xtest,ytest,s) 
    MatsinglePlus[s,j]<-singlePlusfun(xtrain, ytrain,xtest,ytest,s) 
  }
  MatsingleROC[,j]<-matrix(singleROC)
  
  
  
  for (s in (1:(NumVar-1))){
    s<-as.numeric(s)
    k<-CompareISS
    doubleAUC[s,j]<-doubleAUCfun(xtrain, ytrain,xtest,ytest,s,k) 
    doubleROC[[s]]<-doubleROCfun(xtrain, ytrain,xtest,ytest,s,k) 
    MatdoublePlus[s,j]<-doublePlusfun(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCSVM[s,j]<-doubleAUCfunSVM(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCSVMCross[s,j]<-doubleAUCfunSVMCross(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCNB[s,j]<-doubleAUCfunNB(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCRFCross[s,j]<-doubleAUCfunRFCross(xtrain, ytrain,xtest,ytest,s,k)
    
  }
  
  
  MatdoubleROC[,j]<-matrix(doubleROC)
  
  
  #Null Hypothesis ####
  
  #porque pone NumVar-1 si sería NumVar?
  
  # Training set
  
  train$Label<-sample(train$Label)
  test$Label<-sample(test$Label)
  
  # Test set
  
  
  xtrain<-model.matrix(Label~. -1, data = train)
  ytrain<-train$Label
  xtest<-model.matrix(Label~. -1, data = test)
  ytest<-test$Label
  
  xtest<-data.frame(xtest)
  xtrain<-data.frame(xtrain)
  
  
  y<-Betas_select[,NumVar]
  X<-Betas_select[,1:(NumVar-1)]
  
  
  levels(ytrain)[1]<-"0"
  levels(ytrain)[2]<-"1"
  levels(ytest)[1]<-"0"
  levels(ytest)[2]<-"1"
  
  
  source("/Users/lauri/Desktop/R/common_pipeline/AUCFun29-3.R")
  
  multipleAUCNBR[1,j]<-multipleAUCfunNB(xtrain, ytrain,xtest,ytest)
  multipleAUCR[1,j]<-multipleAUCfun(xtrain, ytrain,xtest,ytest)
  
  multipleNBROCR[[j]]<-multipleROCfunNB(xtrain, ytrain,xtest,ytest)
  multipleROCR[[j]]<-multipleROCfun(xtrain, ytrain,xtest,ytest)
  
  
  for (s in (1:(NumVar-1))){
    s<-as.numeric(s)
    singleAUCR[s,j]<-singleAUCfun(xtrain, ytrain,xtest,ytest,s) 
    singleROCR[[s]]<-singleROCfun(xtrain, ytrain,xtest,ytest,s) 
    MatsinglePlusR[s,j]<-singlePlusfun(xtrain, ytrain,xtest,ytest,s) 
  }
  MatsingleROCR[,j]<-matrix(singleROCR)
  
  
  
  for (s in (1:(NumVar-1))){
    s<-as.numeric(s)
    k<-CompareISS
    doubleAUCR[s,j]<-doubleAUCfun(xtrain, ytrain,xtest,ytest,s,k) 
    doubleROCR[[s]]<-doubleROCfun(xtrain, ytrain,xtest,ytest,s,k) 
    MatdoublePlusR[s,j]<-doublePlusfun(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCSVMR[s,j]<-doubleAUCfunSVM(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCSVMCrossR[s,j]<-doubleAUCfunSVMCross(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCNBR[s,j]<-doubleAUCfunNB(xtrain, ytrain,xtest,ytest,s,k) 
    doubleAUCRFCrossR[s,j]<-doubleAUCfunRFCross(xtrain, ytrain,xtest,ytest,s,k)
    
  }
  
  
  MatdoubleROCR[,j]<-matrix(doubleROCR)
  
  
}

if (NISShere==1){
  names2<-sapply(1:(NumVar-1), function(i){paste0("ISS/",names(xtrain)[i])})
}else{
  names2<-sapply(1:(NumVar-1), function(i){paste0("NISS/",names(xtrain)[i])})
}

trial<-NULL

h=1
for (b in (2:NumVar-1)) {
  print(b)
  trial[b]<-names2[h]
  h=h+1
}

doubleAUC<-as.data.frame(doubleAUC)
doubleAUC<-mutate(doubleAUC, Means=rowMeans(doubleAUC))
row.names(doubleAUC)<-trial

singleAUC<-as.data.frame(singleAUC)
singleAUC<-mutate(singleAUC, Means=rowMeans(singleAUC))
row.names(singleAUC)<-names(xtrain)

doubleAUCR<-as.data.frame(doubleAUCR)
doubleAUCR<-mutate(doubleAUCR, Means=rowMeans(doubleAUCR))
row.names(doubleAUCR)<-trial

singleAUCR<-as.data.frame(singleAUCR)
singleAUCR<-mutate(singleAUCR, Means=rowMeans(singleAUCR))
row.names(singleAUCR)<-names(xtrain)

multipleAUCNBR<-as.data.frame(multipleAUCNBR)
multipleAUCNBR["Means"]<-rowMeans(multipleAUCNBR)
multipleAUCR<-as.data.frame(multipleAUCR)
multipleAUCR["Means"]<-rowMeans(as.data.frame(multipleAUCR))

multipleAUCNB<-as.data.frame(multipleAUCNB)
multipleAUCNB["Means"]<-rowMeans(as.data.frame(multipleAUCNB))
multipleAUC<-as.data.frame(multipleAUC)
multipleAUC["Means"]<-rowMeans(as.data.frame(multipleAUC))

Final<-data.frame(MultiNB=t(multipleAUCNB),MultiNBRand=t(multipleAUCNBR),Multi=t(multipleAUC),MultiRand=t(multipleAUCR) )
FinalMeans<-data.frame(MultiNB=multipleAUCNB$Means,MultiNBRand=multipleAUCNBR$Means,Multi=multipleAUC$Means,MultiRand=multipleAUCR$Means )

#sacar ROC CURVES #####

print("here")

for (g in 1:(NumVar-1)){
  
  plot(MatsingleROC[[g,1]],lwd=3,main=paste("ROC curve of", names(xtrain)[g]))
  for (b in 1:N){
    
    plot(MatsingleROCR[[g,b]],col=b,lty=3,add=TRUE)
  }
}


#grid.arrange(plot3[[l]], arrangeGrob(plot4[[l]],plot[[l]], ncol=2), ncol=1)

for (t in 1:(NumVar-1)){
  
  plot(MatdoubleROC[[t,1]],lwd=3,main=paste("ROC curve of", trial[t]))
  for (b in 1:N){
    plot(MatdoubleROCR[[t,b]],col=b,lty=3,add=TRUE)
  }
  
}
#añadir bien las plots de multiple ROC solo hay una!!

plot(multipleROC[[1]],lwd=3,main=paste("ROC curve of", trial[1]))
for (b in 1:N){
  plot(multipleNBROCR[[b]],col=b,lty=3,add=TRUE)
}



#
# #P value of systematic analysis
#
# max1<-(c(rep(0,(NumVar-1))))
# max2<-(c(rep(0,(NumVar-1)*2)))
# 
# for(f in 1:(NumVar-1)){
#   for (h in 1:N){
#     if (singleAUCR[f,h]>singleAUC$Means[f]){
#       max1[f]<-max1[f]+1
#     }
#   }
# }
# 
# for(f in (NumVar-1)){
#   for (h in N){
#     if (doubleAUCR[f,h]>doubleAUC$Means[f]){
#       max2[f]<-max2[f]+1
#     }
#   }
# }
#
#
#

AUCT<-as.data.frame(t(singleAUC[1:dim(singleAUC)[2]-1]))
AUC2T<-as.data.frame(t(singleAUCR[1:dim(singleAUCR)[2]-1]))
one<-as.data.frame(singleAUC$Means)
two<-as.data.frame(singleAUCR$Means)
Mean<-data.frame(one,two)
Mean<-as.matrix(Mean)
#
nm<-names(AUCT)
for (j in 1:(NumVar-1)){
  Mono<-data.frame(Mono=AUCT[,j],Label=as.factor(c(rep("model",dim(AUCT)[1]))))
  Mono1<-data.frame(Mono=AUC2T[,j],Label=as.factor(c(rep("permuted data",dim(AUC2T)[1]))))
  Mono<-rbind(Mono,Mono1)
  print(ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+geom_vline(xintercept =Mean[j,],linetype = 2,color="black",show.legend = TRUE)+labs(title= paste("Permuted versus real model, density plot of AUC curve of",nm[j]),y=paste("Density",nm[j]),x="AUC values")+geom_text(aes(x=as.numeric(Mean[j,1]),y=-0.25),label=signif(Mean[j,1], digits = 2))+geom_text(aes(x=as.numeric(Mean[j,2]),y=-0.25),label=signif(Mean[j,2], digits = 2)))
}

AUCTd<-as.data.frame(t(doubleAUC[1:dim(doubleAUC)[2]-1]))
AUC2Td<-as.data.frame(t(doubleAUCR[1:dim(doubleAUCR)[2]-1]))
three<-as.data.frame(doubleAUC$Means)
four<-as.data.frame(doubleAUCR$Means)
Meand<-data.frame(three,four)
Meand<-as.matrix(Meand)
nmd<-names(AUCTd)
#
#
for (j in 1:(NumVar-1)){
  Monod<-data.frame(Monod=AUCTd[,j],Label=as.factor(c(rep("model",dim(AUCTd)[1]))))
  Mono1d<-data.frame(Monod=AUC2Td[,j],Label=as.factor(c(rep("permuted data",dim(AUC2Td)[1]))))
  Monod<-rbind(Monod,Mono1d)
  print(ggdensity(Monod, x = "Monod", fill = "Label", palette = "jco")+geom_vline(xintercept=Meand[j,],linetype = 2,color="black",show.legend = TRUE)+labs(title= paste("Permuted versus real model, density plot of AUC curve of",nmd[j]),y=paste("Density",nmd[j]),x="AUC values")+geom_text(aes(x=as.numeric(Meand[j,1]),y=-0.25),label=signif(Meand[j,1], digits = 2))+geom_text(aes(x=as.numeric(Meand[j,2]),y=-0.25),label=signif(Meand[j,2], digits = 2)))
}
#
n <- length(plot)
#  nCol <- floor(sqrt(n))
# # do.call("grid.arrange", grobs=c(plots, ncol=nCol))
# # grid.arrange(grobs = plot, ncol = 2)

# ###Accuracy, Precision etc
#
plot3<-list()
plot4<-list()
Data<-c("Accuracy","Sensitivity","Specificity","Precision")


#
#Accuracy
Mono<-NULL
MonoR<-NULL
for (s in 1:(NumVar-1)){
  for (d in 1:4){
    for (j in 1:N){
      Mono1<-data.frame(Mono=MatsinglePlus[s,j][[1]][[d]])
      Mono<-rbind(Mono,Mono1)
      Mono1R<-data.frame(MonoR=MatsinglePlusR[s,j][[1]][[d]])  
      MonoR<-rbind(MonoR,Mono1R)
    }
    MonoLab<-data.frame(Mono=Mono,Label=as.factor(c(rep("model",dim(Mono)[1]))))
    MonoLabR<-data.frame(Mono=MonoR,Label=as.factor(c(rep("permuted data",dim(MonoR)[1]))))
    colnames(MonoLabR)[1]<-c("Mono")
    Means<-data.frame(Mean=mean(MonoLab$Mono),MeanR=mean(MonoLabR$Mono))
    Mono<-rbind(MonoLab,MonoLabR)
    plot3[[d]]<-ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+labs(title= paste("Value : ",Data[d], "for", names(xtrain)[s]),x=paste(names(xtrain)[s]))
    #+geom_vline(xintercept=Means[1,],linetype = 2,color="black",show.legend = TRUE)+geom_text(aes(x=as.numeric(Means[1,1]),y=-0.25),label=signif(Means[1,1], digits = 2))+geom_text(aes(x=as.numeric(Means[1,2]),y=-0.25,color="blue"),label=signif(Means[1,2], digits = 2))
    Mono<-NULL
    MonoR<-NULL
    
  }
  print(grid.arrange(plot3[[2]], plot3[[3]],plot3[[4]],plot3[[1]], ncol=2))
  
}
#plot3<-ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+labs(title= paste("Value studied single: ",Data[d], "for", names(xtrain)[s]))

Mono<-NULL
MonoR<-NULL
for (s in 1:(NumVar-1)){
  for (d in 1:4){
    for (j in 1:N){
      Mono1<-data.frame(Mono=MatdoublePlus[s,j][[1]][[d]])
      Mono<-rbind(Mono,Mono1)
      Mono1R<-data.frame(MonoR=MatdoublePlusR[s,j][[1]][[d]])
      MonoR<-rbind(MonoR,Mono1R)
      
    }
    MonoLab<-data.frame(Mono=Mono,Label=as.factor(c(rep("model",dim(Mono)[1]))))
    MonoLabR<-data.frame(Mono=MonoR,Label=as.factor(c(rep("permuted data",dim(MonoR)[1]))))
    colnames(MonoLabR)[1]<-c("Mono")
    Mono<-rbind(MonoLab,MonoLabR)
    plot4[[d]]<-ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+labs(title= paste("Value : ",Data[d], "for", trial[s]),x=paste(trial[s]))
    Mono<-NULL
    MonoR<-NULL
  }
  print(grid.arrange(plot4[[2]], plot4[[3]],plot4[[4]],plot4[[1]], ncol=2))
  
}


Meanq<-data.frame(multipleAUC$Means,multipleAUCR$Means)
MA<-data.frame(Mono=t(multipleAUC[1:dim(multipleAUC)[2]-1]))
MA["Label"]<-as.factor(c(rep("model",dim(MA)[1])))
MAR<-data.frame(Mono=t(multipleAUCR[1:dim(multipleAUCR)[2]-1]))
MAR["Label"]<-as.factor(c(rep("permuted data",dim(MAR)[1])))
Mono<-rbind(MA,MAR)
pp<-ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+geom_vline(xintercept=Meanq[1,1],linetype = 2,color="black",show.legend = TRUE)+labs(title= paste("AUC curve of GLM Multiple",names1),y="Density",x="AUC values")+geom_text(aes(x=as.numeric(Meanq[1,1]),y=-0.25),label=signif(Meanq[1,1], digits = 2))+geom_text(aes(x=as.numeric(Meanq[1,2]),y=-0.25),label=signif(Meanq[1,2], digits = 2))
print(pp+geom_vline(xintercept=Meanq[1,2],linetype = 2,color="black",show.legend = TRUE))

Meanq<-data.frame(multipleAUCNB$Means,multipleAUCNBR$Means)
MA<-data.frame(Mono=t(multipleAUCNB[1:dim(multipleAUCNB)[2]-1]))
MA["Label"]<-as.factor(c(rep("model",dim(MA)[1])))
MAR<-data.frame(Mono=t(multipleAUCNBR[1:dim(multipleAUCNBR)[2]-1]))
MAR["Label"]<-as.factor(c(rep("permuted data",dim(MAR)[1])))
Mono<-rbind(MA,MAR)
pp<-ggdensity(Mono, x = "Mono", fill = "Label", palette = "jco")+geom_vline(xintercept=Meanq[1,1],linetype = 2,color="black",show.legend = TRUE)+labs(title= paste("AUC curve of NB Multiple",names1),y="Density",x="AUC values")+geom_text(aes(x=as.numeric(Meanq[1,1]),y=-0.25),label=signif(Meanq[1,1], digits = 2))+geom_text(aes(x=as.numeric(Meanq[1,2]),y=-0.25),label=signif(Meanq[1,2], digits = 2))
print(pp+geom_vline(xintercept=Meanq[1,2],linetype = 2,color="black",show.legend = TRUE))

ee<-regex(names,".")
NamesModels<-c("doubleAUCNB","doubleAUCNBR","doubleAUCRFCross","doubleAUCRFCrossR","doubleAUCSVM","doubleAUCSVMR","doubleAUCSVMCross","doubleAUCSVMCrossR")
q<-list(doubleAUCNB,doubleAUCNBR,doubleAUCRFCross,doubleAUCRFCrossR,doubleAUCSVM,doubleAUCSVMR,doubleAUCSVMCross,doubleAUCSVMCrossR)
s<-lapply(q,function(i){MeansNames(i,trial)})
Models<-data.frame(s[[1]][,dim(doubleAUCNB)[2]])
for (h in 1:length(s)){
  Models[,h]<-data.frame(s[[h]][,dim(doubleAUCNB)[2]])
}
row.names(Models)<-trial
names(Models)<-NamesModels

Final<-data.frame(MultiNB=t(multipleAUCNB),MultiNBRand=t(multipleAUCNBR),Multi=t(multipleAUC),MultiRand=t(multipleAUCR) )
FinalMeans<-data.frame(MultiNB=multipleAUCNB$Means,MultiNBRand=multipleAUCNBR$Means,Multi=multipleAUC$Means,MultiRand=multipleAUCR$Means )








