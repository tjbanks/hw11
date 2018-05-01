function seg = getseg(LFP,xcld,minLen)
LFP = reshape(LFP,[],1);
T = length(LFP);

alt0 = diff(LFP==0);
alt = reshape(find(alt0),1,[]);
if alt0(alt(1))==-1
    alt = [0,alt];
end
if alt0(alt(end))==1
    alt = [alt,T];
end
alt = reshape(alt,2,[]);
dur = alt(2,:)-alt(1,:);
alt = alt(:,dur>=xcld);

seg = reshape(alt,1,[]);
if seg(1)==0
    seg = seg(2:end);
else
    seg = [0,seg];
end
if seg(end)==T
    seg(end) = [];
else
    seg = [seg,T];
end
seg = reshape(seg,2,[]);
segdur = seg(2,:)-seg(1,:);
seg = seg(:,segdur>=minLen);
%seg(1,:) = seg(1,:)+1;
end