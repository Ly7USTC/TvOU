# Functions copied from the original scripts (mcapply.R), used only to check
# that the clean code reproduces them. Not part of the package.

exactGammaOU_3<-function(alpha,beta,y_0,T=10,freq=0.001){
  N<-T/freq
  Qu<-1+0.5*sin(5*10)
  y_gammadrive<-rep(0,N)
  y_gammadrive<-c(y_0,y_gammadrive)
  for (i in 2:(N+1)) {
    tau<-freq
    detertrend<-exp(-Qu*tau)*y_gammadrive[i-1]
    tilde_gamma<-rgamma(1,alpha*tau,beta*exp(Qu*tau))
    tilde_N<-rpois(1,1/2*alpha*Qu*tau^2)
    S<-0
    if(tilde_N>0){
     for (j in 1:tilde_N) {
       u<-runif(1)
       S<-S+rexp(1,beta*exp(Qu*tau*sqrt(u)))
     }
    }
    y_gammadrive[i]<-detertrend+tilde_gamma+S
  }
  return(y_gammadrive)
}

appro_TVgammaOU_2<-function(alpha,beta,y_0,T=10,freq=0.001,Qt){
  N<-T/freq
  y_gammadrive<-rep(0,N)
  y_gammadrive<-c(y_0,y_gammadrive)
  for (i in 2:(N+1)) {
    tau<-freq
    detertrend<-exp(-Qt[i-1]*tau)*y_gammadrive[i-1]
    tilde_gamma<-rgamma(1,alpha*tau,beta*exp(Qt[i-1]*tau))
    tilde_N<-rpois(1,1/2*alpha*Qt[i-1]*tau^2)
    S<-0
    if(tilde_N>0){
      for (j in 1:tilde_N) {
        u<-runif(1)
        S<-S+rexp(1,beta*exp(Qt[i-1]*tau*sqrt(u)))
      }
    }
    y_gammadrive[i]<-detertrend+tilde_gamma+S
  }
  return(y_gammadrive)
}

theorectical_gammaOU<-function(alpha,beta,y_0,T,freq){
  N<-T/freq
  Qu<-1+0.5*sin(5*10)
  t_gammaOU<-rep(0,N)
  t_gammaOU<-c(y_0,t_gammaOU)
  for (i in 2:(N+1)) {
    t<-freq*(i-1)
    t_gammaOU[i]<-exp(-Qu*t)*y_0+1/Qu*(1-exp(-Qu*t))*alpha/beta
  }
  return(t_gammaOU)
}

jump_filtering_1storder<-function(y,delta,beta){
  v_n<-delta^beta
  delta_y<-diff(y,1)
  indicator<-(abs(delta_y)<v_n)
  jump_filtered<-delta_y*indicator
  above<-sum(jump_filtered)
  down<-sum(y[-length(y)]*delta)
  return(-above/down)
}

BPV_estimator<-function(y,delta,r){
  delta_y<-diff(y,1)
  abs_deltay<-abs(delta_y)
  BPV<-sum(abs_deltay[-1]^r[2]*abs_deltay[-length(abs_deltay)]^r[1])
  down<-sum(y[-length(y)]*delta)
  return(BPV/down)
}

BPV_estimator_m<-function(y,delta,r){
  n<-length(y)
  delta_y<-diff(y,1)
  abs_deltay<-abs(delta_y)
  BPV<-sum(abs_deltay[-1]^r[2]*abs_deltay[-length(abs_deltay)]^r[1])
  a<-y[-c(n-1,n)]
  b<-y[-c(1,n)]
  down<-delta*sum(a^r[1]*b^r[2])
  return(BPV/down)
}
