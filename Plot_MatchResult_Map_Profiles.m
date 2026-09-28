% =========================================================================
% 文件名：Plot_MatchResult_Map_Profiles.m
% 功能：① 绘制中国区域地图 + MatchResult 匹配点分布
%       ② 随机选取10个点标红，并在右侧绘制对应消光系数廓线
% =========================================================================

% % clc; clear; close all;

%% ===== 0. 加载数据（假设 MatchResult 已在工作区）=====
Data       = double(MatchResult.Data);         % [M, 1299]
ProfileAlti = double(MatchResult.ProfileAlti); % [1291, 1] 单位 km

lon_all = Data(:, 1);   % ERA5_Lon
lat_all = Data(:, 2);   % ERA5_Lat
N_total = size(Data, 1);

%% ===== 1. 随机选取10个点 =====
% rng(40);  % 固定随机种子，保证可复现
Npoint=8;
sel_idx = sort(randperm(N_total, Npoint));   % Npoint个随机行索引（升序）

sel_lon = lon_all(sel_idx);
sel_lat = lat_all(sel_idx);
sel_ext = Data(sel_idx, 9:end);   % [10, 1291]，消光系数廓线

%% ===== 2. 配色方案（学术风格，10色）=====
% 使用 ColorBrewer Set1 风格的10色
cmap10 = [
    0.894  0.102  0.110;   % 红
    0.216  0.494  0.722;   % 蓝
    0.302  0.686  0.290;   % 绿
    0.596  0.306  0.639;   % 紫
    1.000  0.498  0.000;   % 橙
    1.000  1.000  0.200;   % 黄
    0.651  0.337  0.157;   % 棕
    0.969  0.506  0.749;   % 粉
    0.400  0.400  0.400;   % 灰
    0.000  0.749  0.749;   % 青
];

%% ===== 3. 创建画布：1行2列，左宽右窄 =====
fig = figure('Color', 'white', 'Position', [100, 100, 1200, 500]);
t   = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% -----------------------------------------------------------------------
%% ===== Tile 1: 地图 + 匹配点分布 =====
% -----------------------------------------------------------------------
nexttile(1);
ax1 = gca;
hold(ax1, 'on');

% ----- 3.1 绘制中国国界线 -----
% 使用 MATLAB 自带 shaperead 读取国界（需 Mapping Toolbox）
% 若无 Mapping Toolbox，改用下方备用方案
try
    % 方案A：Mapping Toolbox（推荐）
    china = load('J:\6_shp\countryProvince.mat');
    china=china.combined_shp;
    for k = 1 : length(china)
        plot(ax1, china(k).X, china(k).Y, ...
             'k-', 'LineWidth', 0.8);
    end
catch
    warning('未检测到国界数据，跳过国界线绘制。');
end

% ----- 3.2 绘制所有匹配点（浅灰小点）-----
h_all = scatter(ax1, lon_all, lat_all, ...
                8, [0.7 0.7 0.7], 'filled', ...
                'MarkerFaceAlpha', 0.5, ...
                'DisplayName', sprintf('All matched points (N=%d)', N_total));

% ----- 3.3 绘制10个选中点（各自颜色，大圆点+黑边）-----
h_sel = gobjects(Npoint, 1);
for i = 1 : Npoint
    h_sel(i) = scatter(ax1, sel_lon(i), sel_lat(i), ...
                       20, cmap10(i,:), 'filled', ...
                       'MarkerEdgeColor', 'k', ...
                       'LineWidth', 0.8, ...
                       'DisplayName', sprintf('P%02d (%.2f°E, %.2f°N)', ...
                                              i, sel_lon(i), sel_lat(i)));
    % 标注编号
    text(ax1, sel_lon(i)+0.3, sel_lat(i)+0.3, ...
         sprintf('P%02d', i), ...
         'FontSize', 8, 'FontWeight', 'bold', ...
         'Color', cmap10(i,:));
end

% ----- 3.4 坐标轴设置 -----
xlim(ax1, [70, 140]);
ylim(ax1, [3, 54]);
xlabel(ax1, 'Longitude (°E)', 'FontSize', 12);
ylabel(ax1, 'Latitude (°N)',  'FontSize', 12);
title(ax1, 'Spatial Distribution of ERA5–ACDL Matched Points', ...
      'FontSize', 13, 'FontWeight', 'bold');

% 经纬度刻度
set(ax1, 'XTick', 70:10:140, 'YTick', 3:5:54, ...
         'Box', 'on', 'FontSize', 10, 'Layer', 'top', ...
         'GridColor', [0.8 0.8 0.8], 'XGrid', 'on', 'YGrid', 'on');

legend(ax1, [h_all; h_sel], ...
       [{sprintf('All matched (N=%d)', N_total)}, ...
        arrayfun(@(i) sprintf('P%02d', i), 1:Npoint, 'UniformOutput', false)], ...
       'Location', 'southwest', 'FontSize', 8, ...
       'NumColumns', 2, 'Box', 'on');

hold(ax1, 'off');

% -----------------------------------------------------------------------
%% ===== Tile 2: Npoint条消光系数廓线 =====
% -----------------------------------------------------------------------
nexttile(2);
ax2 = gca;
hold(ax2, 'on');

h_prof = gobjects(Npoint, 1);
for i = 1 : Npoint
    ext_profile = sel_ext(i, :);   % [1, 1291]

    % 将无效值（<0 或极大值）替换为 NaN
    ext_profile(ext_profile < 0)    = NaN;
    ext_profile(ext_profile > 100) = NaN;

    h_prof(i) = plot(ax2, ext_profile, ProfileAlti, ...
                     '-', ...
                     'Color',     cmap10(i,:), ...
                     'LineWidth', 1.2, ...
                     'DisplayName', sprintf('P%02d (%.1f°E, %.1f°N)', ...
                                            i, sel_lon(i), sel_lat(i)));
end

% ----- 参考线：零线 -----
xline(ax2, 0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
      'Alpha', 0.7, 'HandleVisibility', 'off');

% ----- 坐标轴设置 -----
ylim(ax2, [ProfileAlti(end), ProfileAlti(1)]);   % 高度范围
% x轴适当留白
ext_all_valid = sel_ext;
ext_all_valid(ext_all_valid < 0 | ext_all_valid > 1e10) = NaN;
xmax = max(ext_all_valid(:), [], 'omitnan');
if isnan(xmax) || xmax <= 0, xmax = 0.1; end
xlim(ax2, [1e-4, 100]);
set(ax2, 'XScale', 'log');

xlabel(ax2, 'Extinction Coefficient at 532 nm (km^{-1})', ...
       'FontSize', 12);
ylabel(ax2, 'Altitude (km)', 'FontSize', 12);
title(ax2, 'Extinction Coefficient Profiles of Selected Points', ...
      'FontSize', 13, 'FontWeight', 'bold');

set(ax2, 'Box', 'on', 'FontSize', 10, ...
         'XGrid', 'on', 'YGrid', 'on', ...
         'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.8);

legend(ax2, h_prof, 'Location', 'northeast', ...
       'FontSize', 9, 'Box', 'on');

hold(ax2, 'off');

%% ===== 4. 总标题 =====
title(t, 'ERA5–ACDL Spatiotemporal Matching Results', ...
      'FontSize', 15, 'FontWeight', 'bold');

%% ===== 5. 导出高分辨率图片 =====
% exportgraphics(fig, 'MatchResult_Map_Profiles.png', 'Resolution', 300);
% disp('Figure saved.');