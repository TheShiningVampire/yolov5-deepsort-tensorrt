#include "deepsort.h"

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger) {
    this->gpuID = gpuID;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = 100;
    this->maxCosineDist = 0.25;
    this->gLogger = gLogger;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID, gLogger);
    int ret = enginePath.find(".onnx");
    if (ret != -1)
        featureExtractor->loadOnnx(enginePath);
    else
        featureExtractor->loadEngine(enginePath);
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets, vector<vector<DetectBox>>& traj)
{
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    result.clear();
    results.clear();
    // end_point.clear();
    // end_points.clear();
    trajectory_point.clear();
    trajectory_points.clear();

    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }

    // traj_ends.clear();
    // for (auto r : end_point) {
    //     DETECTBOX i = r.second;
    //     DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
    //     b.trackID = (float)r.first;
    //     traj_ends.push_back(b);
    // }
    // for (int i = 0; i < end_points.size(); ++i) {
    //     CLSCONF c = end_points[i].first;
    //     traj_ends[i].classID = c.cls;
    //     traj_ends[i].confidence = c.conf;
    // }

    traj.clear();
    for (auto t : trajectory_point) {
        vector<DetectBox> traj_point;
        for (auto r : t) {
            DETECTBOX i = r.second;
            DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
            b.trackID = (float)r.first;
            traj_point.push_back(b);
        }
        traj.push_back(traj_point);
    }

    for (int i = 0; i < trajectory_points.size(); ++i) {
        for (int j = 0; j < trajectory_points[i].size(); ++j) {
            CLSCONF c = trajectory_points[i][j].first;
            traj[i][j].classID = c.cls;
            traj[i][j].confidence = c.conf;
        }
    }
}


void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2)
{
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);

        // Predict trajectory endpoint
        objTracker->traj_predict();


        result.clear();
        results.clear();

        // end_point.clear();
        // end_points.clear();

        trajectory_point.clear();
        trajectory_points.clear();

        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));

            // end_point.push_back(make_pair(track.track_id, track.to_tlwh_traj_pred()));
            // end_points.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh_traj_pred()));

            vector<DETECTBOX> trajectory = track.to_tlwh_traj_pred();
            vector<RESULT_DATA> temp;
            vector<std::pair<CLSCONF, DETECTBOX>> temps;
            for (DETECTBOX i : trajectory) {
                // trajectory_point.push_back(make_pair(track.track_id, i));
                // trajectory_points.push_back(make_pair(CLSCONF(track.cls, track.conf) ,i) );
                temp.push_back(make_pair(track.track_id, i));
                temps.push_back(make_pair(CLSCONF(track.cls, track.conf) ,i) );
            }
            trajectory_point.push_back(temp);
            trajectory_points.push_back(temps);
        }
    }
}

void DeepSort::sort(vector<DetectBox>& dets) {
    DETECTIONS detections;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = r.first;
        dets.push_back(b);
    }
}

void DeepSort::sort(DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}
