function [ p_base, p_min, p_max ] = dayprof( h_prof, pmin_mult )
% dayprof adds noise to each of the household profiles
%   reads in the household profile (h_prof) and fraction of max
%       controllable load to calculate min controllable load (pmin_mult)
%   outputs house profile with [ p_base, p_min, p_max ] representing the
%       baseload in kW, min controllable load in kW, max controllable load 
%       in kW respectively with the noise that should be saved

%% Initialize output variables
p_base = zeros(24,1);
p_max = zeros(24,1);
p_min = zeros(24,1);

%% Set arbitrary noise for output parameters
noise = 3/100; % base load noise fraction
noise1 = 8/100; % max controllable load noise fraction
noise2 = 6/100; % min controllable load noise fraction

%% Calculate ouput data
    for i=1:24
        p_base(i) = (h_prof(i,12) + (noise*rand(1,1)-noise/2))*h_prof(25,12);
        p_max(i) = sum(h_prof(i,2:8).*h_prof(25, 2:8));
        p_min(i) = p_max(i) + sum(h_prof(i,2:8).*h_prof(25, 2:8).*pmin_mult(i,2:8));
        p_max(i) =  p_max(i) +  (p_max(i)*(noise1*rand(1,1)-noise1/2));
        p_min(i) =  p_min(i) +  (p_min(i)*(noise2*rand(1,1)-noise2/2));
    end    

end

