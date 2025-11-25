clear all, close all, clc

gt = load("SanFrancisco_gt.mat").gt;


map = colormap(hsv(max(double(gt(:)))));
map = [0 0 0; map];
imagesc(gt), axis image, impixelinfo, colorbar, colormap(map), axis off

colorbar( 'YTick',linspace(0.5,4.5,6),'YTickLabel', {'Unassigned', 'Bare Soil', 'Mountain', 'Water', 'Urban', 'Vegetation'},'FontSize',25)


