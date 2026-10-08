function handle=op_plot(model,a,path)
% POD + bounds with base plotting; PNG output on old MATLAB.
a=a(:); lo=min(a); hi=max(a)*1.5;
if strcmp(model.x_transform,'log'), grid=exp(linspace(log(lo),log(hi),100))'; else, grid=linspace(lo,hi,100)'; end
handle=figure('Visible','off'); plot(grid,op_pod(model,grid),'LineWidth',2); hold on;
if strcmp(model.kind,'hitmiss')
 [lower,upper]=op_lr_band(model,grid); plot(grid,lower,'--',grid,upper,'--');
else
 probabilities=linspace(.001,.999,200); low=zeros(size(probabilities)); upper=low;
 for i=1:numel(probabilities)
  p=probabilities(i); [center,q]=op_size(model,p); low(i)=op_size_upper(model,p,.95,'wald');
  if strcmp(model.x_transform,'log'), upper(i)=exp(2*q-log(low(i))); else, upper(i)=2*q-low(i); end
 end
 plot(low,probabilities,'--',upper,probabilities,'--');
end
plot([lo,hi],[.9,.9],':'); ylim([0,1]); xlim([lo,hi]);
if strcmp(model.x_transform,'log'), set(gca,'XScale','log'); end
xlabel('Flaw size (input units)'); ylabel('Detection probability'); grid on;
title(sprintf('a90 %.5g; a90/95 %.5g',model.a90,model.a90_95));
if nargin>=3 && ~isempty(path), print(handle,path,'-dpng','-r150'); end
end
