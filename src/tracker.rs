use crate::bbox::BBox;
use crate::detector::Detection;
use crate::sort::{match_detections_to_trackers, KalmanFilter};

#[derive(Clone)]
pub struct TrackedObject {
    pub id: usize,
    pub bbox: BBox,
    pub _class_id: usize,
    pub class_name: String,
    pub confidence: f32,
}

pub struct Track {
    pub id: usize,
    pub kf: KalmanFilter,
    pub time_since_update: usize,
    pub hits: usize,
    pub hit_streak: usize,
    pub age: usize,
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
}

impl Track {
    pub fn new(id: usize, detection: &Detection) -> Self {
        let z = BBox::from_rect(detection.bbox).to_z_vector();
        Self {
            id,
            kf: KalmanFilter::new(z),
            time_since_update: 0,
            hits: 1,
            hit_streak: 1,
            age: 1,
            class_id: detection.class_id,
            class_name: detection.class_name.clone(),
            confidence: detection.confidence,
        }
    }

    pub fn predict(&mut self) -> BBox {
        let state = self.kf.predict();
        self.age += 1;
        
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        
        BBox::from_z_vector(&state)
    }

    pub fn update(&mut self, detection: &Detection) {
        self.time_since_update = 0;
        self.hits += 1;
        self.hit_streak += 1;
        self.class_id = detection.class_id;
        self.class_name = detection.class_name.clone();
        self.confidence = detection.confidence;
        
        let z = BBox::from_rect(detection.bbox).to_z_vector();
        self.kf.update(z);
    }

    pub fn get_state_bbox(&self) -> BBox {
        BBox::from_z_vector(&self.kf.get_state())
    }
}

pub struct Tracker {
    tracks: Vec<Track>,
    next_id: usize,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
}

impl Tracker {
    pub fn new(max_age: usize, min_hits: usize, iou_threshold: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            min_hits,
            iou_threshold,
        }
    }

    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedObject> {
        let mut predicted_bboxes = Vec::with_capacity(self.tracks.len());
        
        // 1. Predict
        for track in self.tracks.iter_mut() {
            predicted_bboxes.push(track.predict());
        }

        // Prepare detection bboxes
        let det_bboxes: Vec<BBox> = detections.iter().map(|d| BBox::from_rect(d.bbox)).collect();

        // 2. Match
        let (matches, unmatched_dets, _unmatched_trks) = match_detections_to_trackers(
            &det_bboxes,
            &predicted_bboxes,
            self.iou_threshold,
        );

        // 3. Update matched
        for (det_idx, trk_idx) in matches {
            self.tracks[trk_idx].update(&detections[det_idx]);
        }

        // 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets {
            let trk = Track::new(self.next_id, &detections[det_idx]);
            self.tracks.push(trk);
            self.next_id += 1;
        }

        // 5. Remove stale tracks
        self.tracks.retain(|t| t.time_since_update <= self.max_age);

        // 6. Return active tracks
        let mut results = Vec::new();
        for track in &self.tracks {
            // A track is returned if it has enough hits or if it's very new and we just saw it
            // Typically SORT requires min_hits frames to be confident
            if track.time_since_update < 1 && (track.hit_streak >= self.min_hits || self.min_hits == 0) {
                results.push(TrackedObject {
                    id: track.id,
                    bbox: track.get_state_bbox(),
                    _class_id: track.class_id,
                    class_name: track.class_name.clone(),
                    confidence: track.confidence,
                });
            }
        }

        results
    }
}
