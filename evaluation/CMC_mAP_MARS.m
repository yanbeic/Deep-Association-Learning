% clear; 
% TODO: ADD model_name HERE: model_name = XXX;
% model_name = 'mobilenet_b64_dal';
addpath(genpath('utils/'));
normalization = 1;
max_pooling = 1; % if 1 use max_pooling else use mean_pooling

track_test = importdata('info/tracks_test_info.mat');
path = strcat('feature/MARS/', model_name, '/test0.mat');
box_feature_test1 = importdata(path);
path = strcat('feature/MARS/', model_name, '/test1.mat');
box_feature_test2 = importdata(path);
box_feature_test = [box_feature_test1 box_feature_test2];
clearvars box_feature_test1 box_feature_test2
if normalization==1
    box_feature_test = normr(box_feature_test')';
end

% do pooling on each tracklet (mean/max-pooling)
video_feat_test = process_box_feat(box_feature_test, track_test, max_pooling); % video features for test (gallery+query)
if normalization==1
    video_feat_test = normr(video_feat_test')';
end

% prepare gallery & query data
query_IDX = importdata('info/query_IDX.mat');  % load pre-defined query index
label_gallery = track_test(:, 3);
label_query = label_gallery(query_IDX);
cam_gallery = track_test(:, 4);
cam_query = cam_gallery(query_IDX);
feat_gallery = video_feat_test;
feat_query = video_feat_test(:, query_IDX);
cam_amount = size(unique(cam_gallery),1); % how many unique cameras

% compute distances
dist_eu = pdist2(feat_query', feat_gallery', 'euclidean');
% evaluate the results
[CMC_eu, map_eu, r1_pairwise, ap_pairwise] = evaluation_mars(dist_eu', label_gallery, label_query, cam_gallery, cam_query, cam_amount);
CMC_eu = CMC_eu*100; 
map_eu = map_eu*100;
sprintf('Euclidean, r1 = %0.4f, r5 = %0.4f, r10 = %0.4f, r20 = %0.4f, mAP = %0.4f ', CMC_eu(1), CMC_eu(5), CMC_eu(10), CMC_eu(20), map_eu)

