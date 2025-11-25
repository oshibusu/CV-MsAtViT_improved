clear all, close all, clc
gt = load("Flevoland_gt.mat").gt;


map = colormap(hsv(max(double(gt(:)))));
map = [0 0 0; map];

imagesc(gt), axis image, impixelinfo,  colormap(map), axis off, colorbar
colorbar('YTick',linspace(0.5,14.5,16),'YTickLabel', {'Unassigned', 'Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed', ...
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', ... 
                        'Wheat 3', 'Barley', 'Buildings'},'FontSize',30)













%% code segment to dsiplay output result of a particular model
clear all, close all, clc

map = colormap(jet(max(double(gt(:)))));
map = [0 0 0; map];
imagesc(outputs), axis image, impixelinfo, colormap(map), axis off
title("Complex CNN - SE", FontSize=20)
colorbar( 'YTick',linspace(0.5,14.5,16),'YTickLabel', {'Unassigned', 'Water', 'Forest', 'Lucerne', 'Grass', 'Rapeseed', ...
                        'Beet', 'Potatoes', 'Peas', 'Stem Beans', 'Bare Soil', 'Wheat', 'Wheat 2', ... 
                        'Wheat 3', 'Barley', 'Buildings'},'FontSize',25)


