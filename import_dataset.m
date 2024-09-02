%% Import the cardiac dataset

function data = cardiac_import_dataset()

data_path = "datasets/";

data.names = {"min vol left ventricle","min vol right ventricle","min vol left atrium","min vol right atrium", ...
              "max vol left atrium","max vol right atrium","half contraction time"};

combined_data = importdata(data_path + "combined.xlsx");

data.X = combined_data.data(:,[1,3]);
data.f = combined_data.data(:,[4,5,6,7,8,9,10]);
data.times = combined_data.data(:,14);

% ensure all times series are registered to same simulation times
tmax = 600; % milliseconds
numt = 600; % number of time grid points
tgrid = linspace(0,tmax,numt);
data.tgrid = tgrid';

for i = 1:size(data.X,1)

    % time series outputs
    dx = data.X(i,1);
    dt = data.X(i,2);
    time_series_path = data_path + "time_series/simulation_" ...
                       + num2str(1000*dx,'%u') + "um_" ...
                       + num2str(dt,'%u') + ".0/";
    f_la = importdata(time_series_path + "la_endo.vol.dat");
    f_lv = importdata(time_series_path + "lv_endo.vol.dat");
    f_ra = importdata(time_series_path + "ra_endo.vol.dat");
    f_rv = importdata(time_series_path + "rv_endo.vol.dat");

    data.ft(i,:,1) = interp1(f_la(:,1), f_la(:,2), tgrid, 'spline'); 
    data.ft(i,:,2) = interp1(f_lv(:,1), f_lv(:,2), tgrid, 'spline'); 
    data.ft(i,:,3) = interp1(f_ra(:,1), f_ra(:,2), tgrid, 'spline'); 
    data.ft(i,:,4) = interp1(f_rv(:,1), f_rv(:,2), tgrid, 'spline'); 

end

% normalisation
data.f = (data.f - mean(data.f,1)) ./ std(data.f,[],1);

% data imputation for the missing computational times
missing = (data.times == 0);
sf = fit([log(data.X(~missing,1)),log(data.X(~missing,2))],log(data.times(~missing)),'poly11');
%plot(sf,[log(data.X(~missing,1)),log(data.X(~missing,2))],log(data.times(~missing)))
data.times(missing) = exp( feval(sf,[log(data.X(missing,1)),log(data.X(missing,2))]) );

end
