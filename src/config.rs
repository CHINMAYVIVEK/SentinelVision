pub struct Config {
    pub video_source: String,
    pub model_path: String,
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub max_age: usize,
    pub min_hits: usize,
    pub iou_threshold: f32,
}

impl Config {
    pub fn default() -> Self {
        Self {
            video_source: "data/input.mp4".to_string(),
            model_path: "models/yolov8.onnx".to_string(),
            conf_threshold: 0.4,
            nms_threshold: 0.4,
            max_age: 30,
            min_hits: 3,
            iou_threshold: 0.3,
        }
    }
}
