use opencv::{core, Result};
use std::fs;

pub struct DebugContext {
    pub debug_dir: String,
}

impl DebugContext {
    pub fn new(debug_dir: &str) -> Result<Self> {
        let _ = fs::create_dir_all(debug_dir); // Ignore error if directory already exists
        Ok(Self {
            debug_dir: debug_dir.to_string(),
        })
    }

    pub fn save_frame(&self, frame: &core::Mat, name: &str) -> Result<()> {
        let path = format!("{}/{}", self.debug_dir, name);
        opencv::imgcodecs::imwrite(&path, frame, &core::Vector::new())?;
        println!("[DEBUG] Saved frame: {}", path);
        Ok(())
    }

    pub fn log_detection(&self, frame_id: u32, detection_idx: usize, bbox_x: i32, bbox_y: i32, bbox_w: i32, bbox_h: i32, confidence: f32, class_name: &str) {
        println!(
            "[DETECTION] Frame {}: #{} | x={} y={} w={} h={} | conf={:.2} | class={}",
            frame_id, detection_idx, bbox_x, bbox_y, bbox_w, bbox_h, confidence, class_name
        );
    }

    pub fn log_detection_summary(&self, frame_id: u32, raw_count: usize, filtered_count: usize, nms_count: usize) {
        println!(
            "[DETECTION_SUMMARY] Frame {}: raw={} -> filtered={} -> nms={}",
            frame_id, raw_count, filtered_count, nms_count
        );
    }

    pub fn log_track(&self, frame_id: u32, track_id: usize, bbox_x: i32, bbox_y: i32, bbox_w: i32, bbox_h: i32, age: usize, hits: usize) {
        println!(
            "[TRACK] Frame {}: ID={} | x={} y={} w={} h={} | age={} hits={}",
            frame_id, track_id, bbox_x, bbox_y, bbox_w, bbox_h, age, hits
        );
    }

    pub fn log_track_summary(&self, frame_id: u32, active_tracks: usize, total_seen: usize) {
        println!(
            "[TRACK_SUMMARY] Frame {}: active={} | total_seen={}",
            frame_id, active_tracks, total_seen
        );
    }

    pub fn log_frame_info(&self, frame_id: u32, width: i32, height: i32, channels: i32) {
        println!(
            "[FRAME_INFO] Frame {}: {}x{} channels={}",
            frame_id, width, height, channels
        );
    }

    pub fn log_render(&self, frame_id: u32, bbox_x: i32, bbox_y: i32, bbox_w: i32, bbox_h: i32, track_id: usize) {
        println!(
            "[RENDER] Frame {}: Drawing ID={} at ({},{}) size {}x{}",
            frame_id, track_id, bbox_x, bbox_y, bbox_w, bbox_h
        );
    }
}
