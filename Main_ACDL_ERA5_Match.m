% =========================================================================
% 文件名：Main_ACDL_ERA5_Match.m
% -------------------------------------------------------------------------
% 【功能】
%   将 ACDL 激光雷达逐廓线消光系数数据与 ERA5 Land 0.1° 网格气象数据进行
%   时空匹配，对落入同一 ERA5 网格（±30min 时间窗口）的多条廓线求平均，
%   输出逐日逐小时匹配样本表。
%
% 【输入】
%   ERA5  : era5.Data[row,col,Ntimes] / era5.Lat[row,col] /
%           era5.Lon[row,col] / era5.Time[Ntimes]（datenum）
%   ACDL  : Extinction_Coefficient_532[1291,N] / Longitude[1,N] /
%           Latitude[1,N] / Profile_UTC_Time[1,N]（TAI93秒）
%
% 【输出】
%   MatchResult.Data     [M × (8+1291)] double
%   MatchResult.VarNames {1 × (8+1291)} cell
%   列顺序: ERA5_Lon, ERA5_Lat, ERA5_Time, ERA5_DewT_K, ERA5_T_K,
%           ERA5_U_ms, ERA5_V_ms, ERA5_SP_Pa, Ext532_L0001~Ext532_L1291
% =========================================================================

% clc; clear; close all;

%% ===================== 路径设置（根据实际修改）=====================
dirERA5  = 'J:\4_ECMWF\ERA5_0.1degree\';   % ERA5 根目录
dirACDL  = 'I:\Data\ACDL\ProfileMat\';  % ACDL .mat 文件目录
OutPath  = 'I:\Data\MatchResult\';       % 输出路径
if ~exist(OutPath, 'dir'), mkdir(OutPath); end

%% ===================== 参数设置 =====================
startDate = datenum('20220601', 'yyyymmdd');
endDate   = datenum('20220602', 'yyyymmdd');

GridRes  = 0.1;           % ERA5 网格分辨率（度）
HalfGrid = GridRes / 2;   % 0.05°，网格半径
nAlt     = 1291;          % ACDL 高度层数

% 时间匹配窗口：±30 分钟
dt_half = 30 / 1440;   % 单位：天

%% ===================== 逐日处理主循环 =====================
prevMonth = -1;  % 用于判断是否需要重新加载月度 ERA5 数据

for dayNum = startDate : 1 : endDate

    DateVec  = datevec(dayNum);
    yearStr  = num2str(DateVec(1), '%04d');
    monthStr = num2str(DateVec(2), '%02d');
    dayStr   = num2str(DateVec(3), '%02d');
    yyyymmdd = [yearStr, monthStr, dayStr];
    yyyymm   = [yearStr, monthStr];

    disp(['Processing: ', yyyymmdd, ' ...']);

    %% ---- Step 1: 按月加载 ERA5 数据（避免每天重复 IO）----
    curMonth = DateVec(1) * 100 + DateVec(2);
    if curMonth ~= prevMonth
        disp(['  Loading ERA5 monthly data: ', yyyymm, ' ...']);

        era5DewT_s = load([dirERA5, yearStr, '\2_metre_dewpoint_temperature\', ...
            yyyymm, '_2_metre_dewpoint_temperature.mat']);
        era5T_s    = load([dirERA5, yearStr, '\2_metre_temperature\', ...
            yyyymm, '_2_metre_temperature.mat']);
        era5U_s    = load([dirERA5, yearStr, '\10_metre_U_wind_component\', ...
            yyyymm, '_10_metre_U_wind_component.mat']);
        era5V_s    = load([dirERA5, yearStr, '\10_metre_V_wind_component\', ...
            yyyymm, '_10_metre_V_wind_component.mat']);
        era5SP_s   = load([dirERA5, yearStr, '\surface_pressure\', ...
            yyyymm, '_surface_pressure.mat']);

        % 统一字段名：era5.Data / era5.Lat / era5.Lon / era5.Time
        ERA5_DewT = era5DewT_s.ECMWF;
        ERA5_T    = era5T_s.ECMWF;
        ERA5_U    = era5U_s.ECMWF;
        ERA5_V    = era5V_s.ECMWF;
        ERA5_SP   = era5SP_s.sp;
        clear era5DewT_s era5T_s era5U_s era5V_s era5SP_s

        %有时候.data的row和col和.Lat/.Lon的row和col不对应。
        if size(ERA5_DewT.Data,1)~=size(ERA5_DewT.Lat,1)
            ERA5_DewT.Data=permute(ERA5_DewT.Data,[2,1,3]);
        end
        if size(ERA5_T.Data,1)~=size(ERA5_T.Lat,1)
            ERA5_T.Data=permute(ERA5_T.Data,[2,1,3]);
        end
        if size(ERA5_U.Data,1)~=size(ERA5_U.Lat,1)
            ERA5_U.Data=permute(ERA5_U.Data,[2,1,3]);
        end
        if size(ERA5_V.Data,1)~=size(ERA5_V.Lat,1)
            ERA5_V.Data=permute(ERA5_V.Data,[2,1,3]);
        end
        if size(ERA5_SP.Data,1)~=size(ERA5_SP.Lat,1)
            ERA5_SP.Data=permute(ERA5_SP.Data,[2,1,3]);
        end


        % 经纬度网格（所有变量一致，取温度场的即可）
        ERA5_Lon2D   = double(ERA5_T.Lon);   % [row, col]
        ERA5_Lat2D   = double(ERA5_T.Lat);   % [row, col]
        [nRow, nCol] = size(ERA5_Lon2D);

        % 构建 KD-Tree（每月重建一次，网格不变则可复用）
        ERA5_Lon_flat = ERA5_Lon2D(:);        % [M, 1]
        ERA5_Lat_flat = ERA5_Lat2D(:);        % [M, 1]
        kdTree = KDTreeSearcher([ERA5_Lon_flat, ERA5_Lat_flat]);

        prevMonth = curMonth;
        disp(['  ERA5 KD-Tree built. Grid size: ', num2str(nRow), 'x', num2str(nCol)]);
    end

    %% ---- Step 2: 提取当天 ERA5 时间索引（逐小时，共24帧）----
    dayTimeStart = dayNum;
    dayTimeEnd   = dayNum + 23/24;
    timeIdx_ERA5 = find(ERA5_T.Time >= dayTimeStart & ERA5_T.Time <= dayTimeEnd);

    if isempty(timeIdx_ERA5)
        disp(['  [WARN] No ERA5 time slice for ', yyyymmdd, ', skip.']);
        continue;
    end

    %% ---- Step 3: 加载当天所有 ACDL 文件并合并 ----
    fileList_ACDL = Fun_filesTraversal(dirACDL, ['ACDL11_', yyyymmdd, '*.mat']);
    fileNum_ACDL  = size(fileList_ACDL, 1);

    if fileNum_ACDL == 0
        disp(['  [WARN] No ACDL file for ', yyyymmdd, ', skip.']);
        continue;
    end

    % 预分配（逐文件追加）
    ACDL_Ext_All  = [];   % [1291, N_total]
    ACDL_Lon_All  = [];   % [1,   N_total]
    ACDL_Lat_All  = [];   % [1,   N_total]
    ACDL_Time_All = [];   % [1,   N_total]  datenum

    for fi = 1 : fileNum_ACDL
        acdl = load(fileList_ACDL{fi, 1});
        acdl=acdl.Output;

        ext   = double(acdl.Extinction_Coefficient_532);  % [1291, N]
        lon   = double(acdl.Longitude(:)');                % [1, N]
        lat   = double(acdl.Latitude(:)');                 % [1, N]

        % Profile_UTC_Time → UTC datenum          
        t_datenum = double(acdl.Profile_UTC_Time(:)')*0.0001/86400+...
            datenum('2000-01-01 12:00:00','yyyy-mm-dd HH:MM:SS');

        ACDL_Ext_All  = [ACDL_Ext_All,  ext];       %#ok<AGROW>
        ACDL_Lon_All  = [ACDL_Lon_All,  lon];       %#ok<AGROW>
        ACDL_Lat_All  = [ACDL_Lat_All,  lat];       %#ok<AGROW>
        ACDL_Time_All = [ACDL_Time_All, t_datenum]; %#ok<AGROW>
    end
    ACDL_altitude=acdl.Altitude;

    N_profiles = size(ACDL_Ext_All, 2);
    disp(['  ACDL profiles loaded: ', num2str(N_profiles), ...
          '  (', num2str(fileNum_ACDL), ' files)']);

    %% ---- Step 4: 空间匹配——为每条廓线找最近 ERA5 网格点 ----
    [nn_idx, nn_dist] = knnsearch(kdTree, [ACDL_Lon_All(:), ACDL_Lat_All(:)]); 
    %nn_idx：每条廓线对应的最近 ERA5 网格点的展平索引（整数）
    %nn_dist：每条廓线到其最近 ERA5 网格点的欧氏距离（单位：度）

    % ACDL只保留真正落入网格内的廓线（欧氏距离 ≤ 对角线半径）
    valid_spatial = (nn_dist <= HalfGrid * sqrt(2));
  
    %% ---- Step 5: 逐小时时间匹配 + 网格内廓线平均 ----
    MatchRows = [];  % 动态累积，最后保存

    for hi = 1 : length(timeIdx_ERA5)
        t_idx = timeIdx_ERA5(hi);
        t_val = ERA5_T.Time(t_idx);   % 当前小时 datenum

        % ACDL时间窗口筛选：±30 min
        valid_time = (abs(ACDL_Time_All - t_val) <= dt_half);

        % ACDL时空双重筛选
        valid_mask = valid_spatial(:) & valid_time(:);
        if ~any(valid_mask), continue; end

        valid_idx    = find(valid_mask);
        valid_nn_idx = nn_idx(valid_mask);   % 对应 ERA5 展平索引

        % 遍历当前小时涉及的每个唯一 ERA5 网格点
        unique_grid_idx = unique(valid_nn_idx);

        for gi = 1 : length(unique_grid_idx)
            gIdx = unique_grid_idx(gi);

            % 落入该网格的廓线索引
            prof_indices = valid_idx(valid_nn_idx == gIdx);

            % 多廓线平均（忽略 NaN）
            ext_profiles = ACDL_Ext_All(:, prof_indices);   % [1291, K]
            ext_mean     = mean(ext_profiles, 2, 'omitnan'); % [1291, 1]

            % 展平索引 → 行列号
            [gRow, gCol] = ind2sub([nRow, nCol], gIdx);

            % 提取对应 ERA5 气象参数
            lon_val  = ERA5_Lon2D(gRow, gCol);
            lat_val  = ERA5_Lat2D(gRow, gCol);
            dewT_val = double(ERA5_DewT.Data(gRow, gCol, t_idx));
            T_val    = double(ERA5_T.Data(gRow, gCol, t_idx));
            U_val    = double(ERA5_U.Data(gRow, gCol, t_idx));
            V_val    = double(ERA5_V.Data(gRow, gCol, t_idx));
            SP_val   = double(ERA5_SP.Data(gRow, gCol, t_idx));

            % 拼接为一行 [1 × (8 + 1291)]
            one_row = [lon_val, lat_val, t_val, ...
                       dewT_val, T_val, U_val, V_val, SP_val, ...
                       ext_mean(:)'];

            MatchRows = [MatchRows; one_row]; %#ok<AGROW>
        end
    end

    %% ---- Step 6: 保存当天结果 ----
    if isempty(MatchRows)
        disp(['  [INFO] No matched rows for ', yyyymmdd, ', skip saving.']);
        continue;
    end

    % 构建列名（8个气象列 + 1291个廓线层）
    ext_varnames = arrayfun(@(k) sprintf('Ext532_Alti_%02d', k), ...
                            1:nAlt, 'UniformOutput', false);
    VarNames = [{'ERA5_Lon','ERA5_Lat','ERA5_Time', ...
                 'ERA5_DewT_K','ERA5_T_K', ...
                 'ERA5_U_ms','ERA5_V_ms','ERA5_SP_Pa'}, ...
                ext_varnames];

    % 保存为结构体 .mat（-v7.3 支持大文件）
    MatchResult.Data     = MatchRows;
    MatchResult.VarNames = VarNames;
    MatchResult.ProfileAlti=ACDL_altitude;
    save([OutPath, 'ACDL_ERA5_Matched_', yyyymmdd, '.mat'], ...
         'MatchResult', '-v7.3');

    disp(['  [OK] Saved ', yyyymmdd, ...
          '  Rows=', num2str(size(MatchRows,1)), ...
          '  Cols=', num2str(size(MatchRows,2))]);

    %是否需要作图。
    plotFlg=1;
    if plotFlg==1
        Plot_MatchResult_Map_Profiles;
    end

    clear MatchRows MatchResult
end

disp('===== All done! =====');