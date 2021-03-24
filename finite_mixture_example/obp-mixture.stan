
data {
  int<lower=1> n; // Number of data
  int<lower=1> p; // Number of clusters
  real y[n];
}


parameters {
  real<lower=0> w[p];
  real<lower=0> r;
  vector<lower=0>[p] sigma;
  vector[p] mu;
}

model {
  
  vector[p] w_abs;
  vector[p] sorted_w_abs;
  vector[p] mu_;
  vector[p] t;
  vector[p] theta;
  vector[p] u;

  int K;
  real threshold;
  
  for (i in 1:p)  {
    w_abs[i] = fabs(w[i]);
  }

  sorted_w_abs = sort_desc(w_abs);



  mu_ = cumulative_sum(sorted_w_abs) - r;
  
  K = 10;

  for (i in 1:p){
    if(sorted_w_abs[i] < (mu_[i]/i)){
      K=(i-1);
      break;
    } 
  }
  
  threshold = mu_[K]/K;
  
  t =  w_abs - threshold;
  
  for (i in 1:p) {
    if(t[i]>0){
      theta[i] = pow(t[i],2);
      }else{
        theta[i]=0;
      } 
  
  }
  u = theta / sum(theta);

  w ~ exponential(1);
  for (k in 1:p){
    target +=inv_gamma_lpdf(sigma[k] | 1, 1);
  }
  
  r ~ exponential(.3);
  
  mu ~ normal(2, 10);

  
  for (i in 1:n){
  vector[p] lps = log(u+.000001); 
  for (k in 1:p){
    lps[k] += normal_lpdf(y[i] | mu[k], sigma[k]);
  }
  target += log_sum_exp(lps);
  }
}





