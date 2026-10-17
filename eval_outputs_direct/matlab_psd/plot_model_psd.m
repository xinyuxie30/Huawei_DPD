clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
dataFile = fullfile(scriptDir, 'model_psd_data.mat');
load(dataFile);

colorRef = [0.45, 0.45, 0.45];
colorError = [0.1216, 0.4667, 0.7059];
colorPred = [0.1725, 0.6275, 0.1725];
colorData = [0.8392, 0.1529, 0.1569];

target_y = target_y(:);
pred_y = pred_y(:);
dataset_error = dataset_error(:);
model_error = model_error(:); % model_error = target_y - pred_y

Len = min(4096, length(target_y));
Len = 2 * floor(Len / 2);
overlap = floor(Len / 2);
window = hann(Len, 'periodic').';

[Ptarget, f] = pwelch(target_y, window, overlap, Len, FsBB, 'centered', 'psd');
[Ppred, ~] = pwelch(pred_y, window, overlap, Len, FsBB, 'centered', 'psd');
[PrefError, ~] = pwelch(dataset_error, window, overlap, Len, FsBB, 'centered', 'psd');
[PmodelError, ~] = pwelch(model_error, window, overlap, Len, FsBB, 'centered', 'psd');

figure('Color', 'w', 'Position', [120, 100, 1100, 600]);
hold on;
plot(f ./ 1e6, 10 * log10(max(PrefError, realmin)), ':', ...
    'Color', colorRef, 'LineWidth', 1.1);
plot(f ./ 1e6, 10 * log10(max(PmodelError, realmin)), ...
    'Color', colorError, 'LineWidth', 1.1);
plot(f ./ 1e6, 10 * log10(max(Ppred, realmin)), '--', ...
    'Color', colorPred, 'LineWidth', 1.1);
plot(f ./ 1e6, 10 * log10(max(Ptarget, realmin)), ...
    'Color', colorData, 'LineWidth', 1.1);
hold off;

ax = gca;
ax.LineWidth = 1.0;
ax.Box = 'on';
ax.XAxis.LineWidth = 1.0;
ax.YAxis.LineWidth = 1.0;
ax.FontSize = 12;
grid on;
ax.GridLineStyle = ':';
ax.GridAlpha = 0.20;

xlabel('Frequency (MHz)', 'FontSize', 14);
ylabel('PSD (dB/Hz)', 'FontSize', 14);
legend('Reference error', 'EA-TCN error (target y - predicted y)', ...
    'Predicted y', 'Dataset y', 'Location', 'northeast');
xlim([-FsBB, FsBB] ./ 2e6);
ylim([-176, -75]);

exportgraphics(gcf, fullfile(scriptDir, 'model_psd_matlab.png'), 'Resolution', 300);
exportgraphics(gcf, fullfile(scriptDir, 'model_psd_matlab.pdf'), 'ContentType', 'vector');
savefig(gcf, fullfile(scriptDir, 'model_psd_matlab.fig'));
