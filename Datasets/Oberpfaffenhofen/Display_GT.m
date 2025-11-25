clear all, close all, clc

gt = load("Oberpfaffenhofen_gt.mat").gt;

figure, 
map = colormap(hsv(max(double(gt(:)))));
map = [0 0 0; map];
imagesc(gt), axis image, impixelinfo, colorbar, colormap(map), axis off

colorbar( 'YTick',linspace(0.5,3.5,5),'YTickLabel', {'Unassigned', 'Build-up Areas', 'Wood Land', 'Open Areas'},'FontSize',15)


