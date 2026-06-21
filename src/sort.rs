use nalgebra::{DMatrix, DVector};
use std::collections::HashSet;

use crate::bbox::BBox;

/// Standard SORT Kalman Filter
pub struct KalmanFilter {
    x: DVector<f32>, // State: [cx, cy, s, r, dcx, dcy, ds]
    p: DMatrix<f32>, // Covariance

    f: DMatrix<f32>, // State transition matrix
    h: DMatrix<f32>, // Measurement matrix
    q: DMatrix<f32>, // Process noise covariance
    r_mat: DMatrix<f32>, // Measurement noise covariance
}

impl KalmanFilter {
    pub fn new(initial_z: DVector<f32>) -> Self {
        let ndim = 4;
        let dt = 1.0;

        let mut f = DMatrix::identity(7, 7);
        for i in 0..3 {
            f[(i, i + ndim)] = dt;
        }

        let mut h = DMatrix::zeros(ndim, 7);
        for i in 0..ndim {
            h[(i, i)] = 1.0;
        }

        let std_weight_position = 1.0 / 20.0;
        let std_weight_velocity = 1.0 / 160.0;

        let mut x = DVector::zeros(7);
        for i in 0..ndim {
            x[i] = initial_z[i];
        }

        let mut p = DMatrix::identity(7, 7);
        for i in 0..3 {
            p[(i, i)] = 10.0;
            p[(i + ndim, i + ndim)] = 1000.0;
        }
        p[(3, 3)] = 10.0;

        let mut q = DMatrix::identity(7, 7);
        for i in 0..3 {
            q[(i, i)] = std_weight_position * std_weight_position;
            q[(i + ndim, i + ndim)] = std_weight_velocity * std_weight_velocity;
        }
        q[(3, 3)] = 1e-2;

        let mut r_mat = DMatrix::identity(ndim, ndim);
        for i in 0..ndim {
            r_mat[(i, i)] = std_weight_position * std_weight_position;
        }

        Self { x, p, f, h, q, r_mat }
    }

    pub fn predict(&mut self) -> DVector<f32> {
        self.x = &self.f * &self.x;
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
        self.x.clone()
    }

    pub fn update(&mut self, z: DVector<f32>) {
        let y = &z - &self.h * &self.x;
        let s = &self.h * &self.p * self.h.transpose() + &self.r_mat;
        
        // Use regularized inversion with small epsilon to improve numerical stability
        let epsilon = 1e-6;
        let mut s_regularized = s.clone();
        for i in 0..s_regularized.nrows() {
            s_regularized[(i, i)] += epsilon;
        }
        
        let s_inv = s_regularized.try_inverse()
            .unwrap_or_else(|| DMatrix::identity(4, 4));
        let k = &self.p * self.h.transpose() * s_inv;
        
        self.x = &self.x + &k * y;
        
        let i = DMatrix::<f32>::identity(7, 7);
        self.p = (i - &k * &self.h) * &self.p;
    }

    pub fn get_state(&self) -> DVector<f32> {
        self.x.clone()
    }
}

pub fn match_detections_to_trackers(
    detections: &[BBox],
    trackers: &[BBox],
    iou_threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    // Greedy IoU-based matching for detection-to-tracker assignment.
    // 
    // This uses a greedy approach: sort all detection-tracker pairs by IoU 
    // in descending order, then greedily match highest IoU pairs first.
    // 
    // For near-optimal assignment (especially under heavy occlusion), 
    // a Hungarian algorithm could replace this. However, greedy matching 
    // is fast, maintains low latency, and works well in practice for 
    // real-time multi-object tracking.
    // 
    // TODO (optimization): Implement Hungarian algorithm for optimal assignment
    // if ID switches or assignment failures are observed in production.
    
    if trackers.is_empty() {
        return (
            vec![],
            (0..detections.len()).collect(),
            vec![],
        );
    }

    if detections.is_empty() {
        return (
            vec![],
            vec![],
            (0..trackers.len()).collect(),
        );
    }

    let mut iou_matrix = vec![vec![0.0; trackers.len()]; detections.len()];
    for (d, det) in detections.iter().enumerate() {
        for (t, trk) in trackers.iter().enumerate() {
            iou_matrix[d][t] = det.iou(trk);
        }
    }

    // Greedy matching (sort all pairs by IoU descending)
    let mut matches = Vec::new();
    let mut unmatched_detections = HashSet::new();
    for i in 0..detections.len() {
        unmatched_detections.insert(i);
    }
    let mut unmatched_trackers = HashSet::new();
    for i in 0..trackers.len() {
        unmatched_trackers.insert(i);
    }

    let mut pairs = Vec::new();
    for d in 0..detections.len() {
        for t in 0..trackers.len() {
            let iou = iou_matrix[d][t];
            if iou >= iou_threshold {
                pairs.push((d, t, iou));
            }
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    for (d, t, _iou) in pairs {
        if unmatched_detections.contains(&d) && unmatched_trackers.contains(&t) {
            matches.push((d, t));
            unmatched_detections.remove(&d);
            unmatched_trackers.remove(&t);
        }
    }

    let unmatched_detections_vec = unmatched_detections.into_iter().collect();
    let unmatched_trackers_vec = unmatched_trackers.into_iter().collect();

    (matches, unmatched_detections_vec, unmatched_trackers_vec)
}
