setwd("/Users/maoran/OneDrive - University of Florida/OneBallPrior-master")
require("rstan")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


y <- c(rnorm(300,0,1),rnorm(300,4,1),rnorm(400,6,1))
n = 1000
p = 10
input_data= list(n=n,p=p,y=y)
l1ball_fit <- stan(file='obp-mixture.stan',
                   data=input_data,
                   chains = 1,
                   control=list(adapt_delta=0.6, max_treedepth=10),
                   iter =11000, warmup = 10000)

mu<- extract(l1ball_fit,"mu", permuted = FALSE)
w<- extract(l1ball_fit,"w", permuted = FALSE)
sigma<- extract(l1ball_fit,"sigma", permuted = FALSE)

r<- extract(l1ball_fit,"r", permuted = FALSE)


proj_l1_ball<- function(x,r){
  
  sorted_abs_x = sort(abs(x),decreasing = TRUE)
  
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
  th = t*(t>0)
  return(abs(th)) 
}
u<- matrix(0,1000,p)
for (i in c(1:1000)){
  theta = proj_l1_ball(w[i,,],r[i,,])
  u[i,] = theta/sum(theta)
}

K <- apply(u,1,function(x) sum(x>1E-5))

hist(K)


