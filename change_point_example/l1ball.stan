functions { 
  real signnum(real x){ 
    return x/fabs(x);
    } 
}

data {
  int<lower=1> n; // Number of data
  int<lower=1> p; // Number of covariates
  matrix[n, p] X;
  real y[n];
}


parameters {
  vector[p] beta;
  real<lower=0> r;
  real<lower=0> sigma;
}



model {
  
  vector[p] beta_abs;
  vector[p] sorted_beta_abs;
  vector[p] mu;
  vector[p] t;
  vector[p] s;
  vector[p] theta;

  int K;
  real threshold;
  
  for (i in 1:p)  {
    beta_abs[i] = fabs(beta[i]);
  }

  sorted_beta_abs = sort_desc(beta_abs);
  
  mu = cumulative_sum(sorted_beta_abs) - r;
  
  for (i in 1:p){
    if(sorted_beta_abs[i] < (mu[i]/i)){
      K=(i-1);
      break;
    } 
  }
  
  threshold = mu[K]/K;
  
  t =  beta_abs - threshold;
  
  for (i in 1:p) {
    s[i]= signnum(beta[i]);
    
    if(t[i]>0){
      theta[i] = t[i]*s[i];
      }else{
        theta[i]=0;
      } 
  
  }


  beta ~ normal(0, 10);
  sigma ~ normal(0, 1);
  r ~ exponential(1);

  y ~ normal(X * theta, sigma);
}
