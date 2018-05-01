prelen = 50;            % Number of passed time samples (50 ms) for input
forward = 10;           % Number of time samples forward to predict (10 ms)
filt_xcld = 50;         % Duration at the edge of segments to be excluded (50 ms)
DetectWindow = 50;      % Detection window (50 ms width)
chunksize = prelen+forward+DetectWindow-1;
truth_thresh = 0.349;	% Ground truth detection threshold
Nconsc = 3;             % Number of consecutive peaks
mindist = 10;           % Minimum distance between two peaks
close all
rng(111);

fs = 1000;
oscBand=[64,84];        % Filter band
[bFilt, aFilt] = butter(2,oscBand/(fs/2));
%load(fullfile('dataset','subject4.mat'));
load('subject4.mat');
T = length(LFP);

% Get start time points of each chunk
xcld = 200;             % Exclude 0 periods more than 200 ms
minLen = 1000;          % Minimum length for a segment
seg = getseg(LFP,xcld,minLen);  % get segment time
nseg = size(seg,2);     % number of segments
ts = cell(1,nseg);      % start time point of chunks
valid = false(1,T);     % mark for valid segments
for i = 1:nseg
    ts{i} = seg(1,i)+1+filt_xcld:seg(2,i)-filt_xcld-chunksize+1;
    valid(seg(1,i)+1:seg(2,i)) = true;
end
ts = cell2mat(ts);
Ttot = length(ts);

% Select training samples
prop = 0.1;                 % Proportion of training samples (10%)
ntrain = round(Ttot*prop);  % number of training samples
t = sort(randsample(ts,ntrain));

% filtfilt for ground truth. causal filter for training input.
ZS = (LFP-mean(LFP(valid)))/std(LFP(valid));    % z-score
ZS_gamma = filtfilt(bFilt,aFilt,ZS);
ZS_causal = filter(bFilt,aFilt,ZS);

x = zeros(ntrain,prelen+2); % training set (prelen inputs,2 outputs)
progress = 0;
for i = 1:ntrain
    x(i,1:prelen) = ZS_causal(t(i):t(i)+prelen-1);
    detectwindow = ZS_gamma(t(i)+prelen+forward-1:t(i)+chunksize-1);
    [alarm,t1pk] = DetectBurst(detectwindow,truth_thresh,Nconsc,mindist);
    if alarm
        t1pk = forward+t1pk-1;          % first peak time relative to current time
    else
        t1pk = forward+DetectWindow;	% if no burst, 1st peak time put to the end of the window.
    end
    x(i,prelen+1) = alarm;
    x(i,prelen+2) = t1pk;
    if progress~=round(i/ntrain*100)
        progress = round(i/ntrain*100);
        disp([num2str(progress),'% completed.']);
    end
end
disp([num2str(sum(x(:,prelen+1))),' out of ',num2str(ntrain),' samples have burst.']);
% csvwrite('train_sub4.csv',x);	% create training set file

figure
hold on
plot(ZS_gamma,'b');
iburst = find(x(:,prelen+1));
tpk = t(iburst)+prelen-1+x(iburst,end)';
plot(tpk,ZS_gamma(tpk),'r.');
plot([1,T],[1,1]*truth_thresh,'g');
axis tight;
legend('filtfilt signal','1st burst peaks','detection threshold');
