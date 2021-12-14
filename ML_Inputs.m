close all
clear
clc

%% Read in raw data
% file with raw data created based on online research (see report for details)
filename = 'Raw_HouseholdProfileData.xlsx'; 

% set variables for each of the sheet names in the input file
sheet_pmin = 'P_ctrl_min'; % fraction of max power controllable loads
sheet_h1 = 'Type1_4 Occupants'; % profile data type 1
sheet_h2 = 'Type2_6 Occupants'; % profile data type 2
sheet_h3 = 'Type3_2 Occupants'; % profile data type 3
sheet_h4 = 'Type4_1 Occupant'; % profile data type 4
sheet_h5 = 'Type5_3 Occupants'; % profile data type 5
sheet_prof = 'HomeType Profile'; % read in specifics of each profile type

% read in data from file
pmin_mult = xlsread ('Raw_HouseholdProfileData.xlsx', 'P_ctrl_min'); 
h1_prof = xlsread (filename, sheet_h1); % read profile data type 1
h2_prof = xlsread (filename, sheet_h2); % read profile data type 2
h3_prof = xlsread (filename, sheet_h3); % read profile data type 3
h4_prof = xlsread (filename, sheet_h4); % read profile data type 4
h5_prof = xlsread (filename, sheet_h5); % read profile data type 5
home_profile = xlsread (filename, sheet_prof); % read profile specifics

h = struct; % struct for all household data compiled

%% Initalize and store household raw data
numdays = 100; % number of days to generate data for
numhomes = home_profile (6,6); % cell that contains the total number of homes in raw data
h.time = zeros(24*numdays, numhomes); % init time of data point in hours
h.pbase = zeros(24*numdays, numhomes); % init baseload power in kW
h.pmin = zeros(24*numdays, numhomes); % init min controllable load in kW
h.pmax = zeros(24*numdays, numhomes); % init max controllable load in kW

for i=1:numhomes 
    % figure out next home profile
   if home_profile (1,6)>0
      h_prof = h1_prof;
      h.homeprof(i) = 1;
      home_profile (1,6) = home_profile (1,6) -1;
   elseif home_profile (2,6)>0
      h_prof = h2_prof;
      h.homeprof(i) = 2;
      home_profile (2,6) = home_profile (2,6) -1;
   elseif home_profile (3,6)>0
      h_prof = h3_prof;
      h.homeprof(i) = 3;
      home_profile (3,6) = home_profile (3,6) -1;
   elseif home_profile (4,6)>0
       h_prof = h4_prof;
       h.homeprof(i) = 4;
       home_profile (4,6) = home_profile (4,6) -1;
   else
       h_prof = h5_prof;
       h.homeprof(i) = 5;
   end
   for j=1:numdays
       time = (j-1)*24 +(1:24);
       h.time(time,i) = time;
      
       [pbase, pmin, pmax] = dayprof(h_prof, pmin_mult); 
       h.pbase(time,i) = pbase;
       h.pmax(time,i) =  pmax;      
       h.pmin(time,i) =  pmin;
       

   end
    
end

%% Calculate total system level data

tot_pmax = zeros(size(h.pmax,1),1);
tot_pmin  = zeros(size(h.pmin,1),1);
tot_pbase = zeros(size(h.pbase,1),1);
for k=1:size(h.pmax,1)

    tot_pmax(k) = sum(h.pmax(k,:));
    tot_pmin(k) = sum(h.pmin(k,:));
    tot_pbase(k) = sum(h.pbase(k,:));
end

%% Produce figures
fig=0;
fig=fig+1;
figure(fig);
axis_font=20;
whitebg(fig,'white');
axis on;
grid on;
set(gca,'FontSize',axis_font,'FontName','Times New Roman');
set(gcf,'Color',[1 1 1]);
hold on;

plot (h.time(:,1), tot_pmax, 'b', 'LineWidth',2, 'DisplayName','P max DR')
hold on
plot (h.time(:,1), tot_pmin , 'r', 'LineWidth',2, 'DisplayName','P min DR')
hold on
plot (h.time(:,1), tot_pmax-tot_pbase , 'g', 'LineWidth',2, 'DisplayName','P base')

xlabel ('Time Stamp [hours]', 'fontsize', axis_font);
ylabel ('Total System Load [kW]','fontsize', axis_font);
xticks((0:24*7:24*numdays))
legend('show','Location','northeast');


%% Output file to be used as input file for Python script

% save all important segments of struct to vectors
htime = h.time; % time stamp in hours
hpmax = h.pmax; % max controllable load in kW for each household
hpmin = h.pmin; % min controllable load in kW for each household
hpbase = h.pbase; % baseload in kW for each household

xlswrite('InputData_HouseholdStates.xls', htime, 'time')
xlswrite('InputData_HouseholdStates.xls', hpmax, 'pmax')
xlswrite('InputData_HouseholdStates.xls', hpmin, 'pmin')
xlswrite('InputData_HouseholdStates.xls', hpbase, 'pbase')