function handle=op_plot_preprocess(result,path)
% A-scan comparison: for multi-trace arrays display first trace, time last.
x=result.raw; y=result.processed; e=result.envelope; threshold=result.cfar_threshold;
if isvector(x), x=x(:)'; y=y(:)'; e=e(:)'; threshold=threshold(:)';
else, n=size(x,ndims(x)); x=reshape(x,[],n); y=reshape(y,[],n); e=reshape(e,[],n); threshold=reshape(threshold,[],n); end
handle=figure('Visible','off'); subplot(2,1,1); plot(x(1,:)); hold on; plot(y(1,:)); legend('raw','processed'); ylabel('RF amplitude'); grid on;
subplot(2,1,2); plot(e(1,:)); hold on; plot(threshold(1,:)); legend('envelope','CFAR threshold'); xlabel('Sample (MATLAB 1 based)'); grid on;
if nargin>1, print(handle,path,'-dpng','-r150'); end
end
