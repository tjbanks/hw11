function [ alarm,t_1st_pk,ncycle,tpeak ] = DetectBurst( X,thresh,Nconsc,mindist )
% mindist = 10;       % Minimum interval between peaks (10 ms)
% thresh = 0.352;     % Detect amplitude threshold
% Nconsc = 3;         % Number of consecutive peaks(>threshold) required

X = reshape(X,1,[]);    % Filtered signal in detection window
d2X = [0,-diff(sign(diff(X)))];     % 2 or [1,1] => peak, -2 or [-1,-1] => trough
tpk = find(d2X==2 | d2X==1&[d2X(2:end),0]==1);	% Peak timing

% Eliminate peaks with interval<mindist
npk = length(tpk);  % number of peaks
intvl = [diff(tpk),0];
idx = 1;
while idx<npk
    if intvl(idx)>=mindist
        idx = idx+1;
        continue;
    end
    if X(tpk(idx))<X(tpk(idx+1))
        tpk(idx) = [];
        intvl(idx) = [];
    else
        tpk(idx+1) = [];
        intvl(idx) = intvl(idx)+intvl(idx+1);
        intvl(idx+1) = [];
    end
    npk = npk-1;
end

% Check whether Nconsc consecutive peaks are over threshold
nthresh = numel(thresh);
alarm = false(size(thresh));    % Alarm for burst detection
tpeak = [];                     % Timing of peaks in burst
if npk>=Nconsc
    for j = 1:nthresh
        t_1st_pk = 0;	% Timing of the first peak
        ncycle = 0;     % Maximum number of consecutive cycles
        overthresh = X(tpk)>thresh(j);
        for i = 1:npk
            if overthresh(i)
                if ~ncycle
                    idx = i;
                end
                ncycle = ncycle+1;
                if ncycle==Nconsc
                    t_1st_pk = tpk(idx);
                    alarm(j) = true;
                end
            elseif alarm(j)
                break;
            else
                ncycle = 0;
            end
        end
    end
    if alarm(j)
        tpeak = tpk(idx:idx+ncycle-1);  % Timing of peaks in burst
    end
else
    t_1st_pk = 0;   % Timing of the first peak
    ncycle = 0;     % Maximum number of consecutive cycles
end

end