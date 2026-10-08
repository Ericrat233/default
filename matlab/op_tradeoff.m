function values=op_tradeoff(model,noise,thresholds)
% Noise/model MUST use same feature and response units.
thresholds=thresholds(:); values=NaN(numel(thresholds),5);
for i=1:numel(thresholds)
 local=model; local.threshold=thresholds(i); decibels=NaN;
 if thresholds(i)>0, decibels=20*log10(thresholds(i)); end
 values(i,:)=[thresholds(i),decibels,noise.pfa_at(thresholds(i)),op_size(local,.9),op_size_upper(local,.9,.95)];
end
end
