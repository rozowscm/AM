function [ap,bp]=TwoModelFigures(x,a,b,p,tlab)
pdirichlet=@(x,a)exp(gammaln(sum(a))-sum(gammaln(a))+log(x+1e-30)*(a(:)-1)); 
q=length(p);
% prior
z0=pdirichlet(x,a);
plot(x(:,1),z0,'k:','linewidth',2.5)
% exact posterior
e=eye(q);
pi=normalizeP(p.*a);   % mixture weights on posterior 
z1=0;
for i=1:q
  z1=z1+pi(i)*pdirichlet(x,a+e(:,i));
end
hold on
plot(x(:,1),z1,'k-','linewidth',2)
% approximate posterior
%ap=modelweightsupdatesim(p',a',0)';
ap=modelweightsupdatesim(p,a,1);
z2=pdirichlet(x,ap);
plot(x(:,1),z2,'k--','linewidth',2.5)
plot(b(1),0,'k^','markersize',10,'MarkerFaceColor','k')
bp=normalizeP(p.*b);
h=plot(bp(1),0,'ko','markersize',10,'MarkerFaceColor','k');
hold off
ylim([0 1.8])
xlabel('$$w_1$$')
title(tlab)
legend('prior','exact posterior','approximate posterior',...
    'location','southoutside','orientation','horizontal');
 
disp('Expected model weights for prior, posterior and approximate posterior and a & a+')  
disp([a/sum(a) (a+pi)/(sum(a)+1) ap/sum(ap) a ap] ) 

text(0.4,1.55,sprintf(' $$\\alpha ~~ = [%1.3f, %1.3f]$$',a(1),a(2)));
text(0.4,1.4,sprintf('$$\\alpha^+ = [%1.3f, %1.3f]$$',ap(1),ap(2)));
