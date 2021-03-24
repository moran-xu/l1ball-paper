library(horseshoe)
library(rstan)
#ftse100 <- read.csv("ftse.csv",header=FALSE)
#cum <- cumsum(ftse100$V1)[700:1000]
#n = length(cum)
n <- 100
p <- 100
X <- matrix(1, ncol = n, nrow = n)
X[upper.tri(X)]=0
mu <- c(rep(30,20),rep(10,20),rep(40,40),rep(20,20))
y <- rnorm(100, mu, sd=3.3)


horse <- horseshoe(y, X, nmc = 3000, method.tau ="halfCauchy", method.sigma ="Jeffreys")
theta_horse <- horse$BetaSamples
theta_mean <- apply(theta_horse,mean,MARGIN = 1)
plot(X%*%theta_mean, type='l')

plot(y,pch=20)
lines(mu,lty=2)

post <- function(beta){
  sum(y-X%*%beta)^2
}

RightCI <- apply(X%*%horse$BetaSamples, 1, function(x) quantile(x,.9))
LeftCI <- apply(X%*%horse$BetaSamples, 1, function(x) quantile(x,.1))
plot(apply(X%*%horse$BetaSamples, 1, function(x) quantile(x,.5)), type = 'l')
polygon(c(1:n, rev(1:n)), c(RightCI ,rev(LeftCI )), col = rgb(1, 0, 0,0.3) )
lines(mu,lty=2)


l1ball <- function(n,p,X,y){
  
  input_data= list(n=n,p=p,X=X,y=y)
  
  l1ball_fit <- stan(file='l1ball.stan',
                     data=input_data,
                     chains = 1,
                     control=list(adapt_delta=0.6, max_treedepth=10),
                     iter =4000, warmup = 3000
  )
  
  beta<- extract(l1ball_fit,"beta", permuted = FALSE)
  r<- extract(l1ball_fit,"r", permuted = FALSE)
  #lam <- extract(l1ball_fit,"lam", permuted = FALSE)
  
  
  
  proj_l1_ball<- function(x,r){
    
    sorted_abs_x = sort(abs(x),decreasing = T)
    
    mu = cumsum(sorted_abs_x) - r
    
    p=length(x)
    
    
    for(i in 1:p){
      if (sorted_abs_x[i]< mu[i]/i){
        K=i-1
        break
      }
      if(i==p){
        K=p
      }
    }
    
    threshold = mu[K]/K
    t = abs(x)- threshold
    sign(x)* t*(t>0)
    
  }
  
  
  theta<- matrix(0,p,1000)
  
  for(i in c(1:1000)){
    theta[,i] = proj_l1_ball(beta[i,,],r[i,,])
  }
  return(theta)
}
RightCI <- apply(X%*%theta, 1, function(x) quantile(x,.9))
LeftCI <- apply(X%*%theta, 1, function(x) quantile(x,.1))
plot(apply(X%*%theta, 1, function(x) quantile(x,.5)), type = 'l')
polygon(c(1:n, rev(1:n)), c(RightCI ,rev(LeftCI )), col = rgb(1, 0, 0,0.3) )
lines(mu,lty=2)


plot_band <- function(beta){
  index <- apply(beta,2,post) < quantile(apply(beta,2,post),.95)
  Betaeff <- beta[,index]
  XBetaSort <- apply(X%*%Betaeff, 1, sort, decreasing = F)
  alpha=.05
  effsamp =length(XBetaSort[,1])
  left <- floor(alpha * effsamp/2)
  right <- ceiling((1 - alpha/2) * effsamp)
  mid <- floor(.5*effsamp)
  left.points <- XBetaSort[left, ]
  right.points <- XBetaSort[right, ]
  plot(y, pch=20,col = rgb(0,0,0,0.5))
  XBmid <- X%*%beta[,which.min(apply(beta,2,post))]
  Xleft <- X%*%beta[,which.max(apply(beta,2,post))]
  Xright <- X%*%beta[,order(apply(beta,2,post),decreasing = TRUE)[2]]
 polygon(c(1:n, rev(1:n)), c(right.points,rev(left.points )), col = rgb(.1, .8, .8,.5), new=TRUE, border=FALSE)
 lines(X%*%beta[,which.min(apply(beta,2,post))], col = 'red', lwd = 5)
  lines(mu,lty=2,lwd=4)
  }
plot_band(theta)

plot_band(horse$BetaSamples)

theta <- l1ball(n,p,X,y)

frechet_mean <- function(theta){
  n <- dim(theta)[1]
  s <- numeric(n)
  for (i in 1:n){
    for (j in 1:n){
      s[i] = s[i] + sum((theta[i,]-theta[j,])^2)
    }
  }
  return(theta[which.min(s),])
}
