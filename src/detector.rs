use opencv::{
    core::{self, Rect},
    prelude::MatTraitConst,
};
use ort::{session::Session, value::Tensor, inputs};
use ndarray::Array4;

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: Rect,
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}

pub struct YoloDetector {
    session: Session,
    conf_threshold: f32,
    nms_threshold: f32,
}

impl YoloDetector {
    pub fn new(
        model_path: &str,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(Self {
            session,
            conf_threshold,
            nms_threshold,
        })
    }

    pub fn detect(&mut self, frame: &core::Mat) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let original_size = frame.size()?;
        let original_w = original_size.width as f32;
        let original_h = original_size.height as f32;

        // ---- Letterbox to 640x640 ----
        let input_w = 640.0;
        let input_h = 640.0;
        let scale = f32::min(input_w / original_w, input_h / original_h);
        let new_w = (original_w * scale).round() as i32;
        let new_h = (original_h * scale).round() as i32;
        let dx = (input_w - new_w as f32) / 2.0;
        let dy = (input_h - new_h as f32) / 2.0;

        // Resize
        let mut resized = core::Mat::default();
        opencv::imgproc::resize(
            frame,
            &mut resized,
            core::Size::new(new_w, new_h),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        // Create a 640x640 image with gray background using copyMakeBorder
        let top = dy.round() as i32;
        let bottom = (input_h - new_h as f32 - dy).round() as i32;
        let left = dx.round() as i32;
        let right = (input_w - new_w as f32 - dx).round() as i32;

        let mut letterboxed = core::Mat::default();
        opencv::core::copy_make_border(
            &resized,
            &mut letterboxed,
            top,
            bottom,
            left,
            right,
            opencv::core::BORDER_CONSTANT,
            core::Scalar::new(114.0, 114.0, 114.0, 0.0),
        )?;

        // ---- BGR -> RGB, normalize to [0, 1], create tensor ----
        let mut tensor_data = vec![0.0f32; 3 * 640 * 640];

        for y in 0..640usize {
            for x in 0..640usize {
                let bgr_pixel = *letterboxed.at_2d::<core::Vec3b>(y as i32, x as i32)
                    .map_err(|_| format!("Failed to read pixel at ({}, {})", x, y))?;
                let idx = y * 640 + x;

                // BGR -> RGB conversion
                tensor_data[idx] = bgr_pixel[2] as f32 / 255.0;
                tensor_data[640 * 640 + idx] = bgr_pixel[1] as f32 / 255.0;
                tensor_data[2 * 640 * 640 + idx] = bgr_pixel[0] as f32 / 255.0;
            }
        }

        // ---- Create input tensor [1, 3, 640, 640] ----
        let input_array = Array4::<f32>::from_shape_vec((1, 3, 640, 640), tensor_data)
            .map_err(|e| format!("Tensor reshape failed: {}", e))?;

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;

        // ---- Run inference ----
        let outputs = self.session.run(inputs![input_tensor])
            .map_err(|e| format!("Inference failed: {}", e))?;

        // ---- Parse output ----
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output extraction failed: {}", e))?;
            
        println!("[DETECTOR] Output shape: {:?}, data len: {}", shape, data.len());
        
        // Determine the shape layout
        let shape_slice: &[i64] = &shape;
        let (num_detections, is_transposed) = if shape_slice == &[1i64, 8400i64, 84i64] {
            (8400, false)
        } else if shape_slice == &[1i64, 84i64, 8400i64] {
            (8400, true)
        } else {
            return Err(format!("Unexpected output shape: {:?}", shape).into());
        };
        
        println!("[DETECTOR] Number of predictions: {}, transposed: {}", num_detections, is_transposed);
        
        // Adjust coordinates for letterbox
        let scale_inv = 1.0 / scale;
        let dx_f32 = dx;
        let dy_f32 = dy;

        let mut bboxes = core::Vector::<Rect>::new();
        let mut confidences = core::Vector::<f32>::new();
        let mut class_ids = Vec::new();

        // Parse each detection
        for i in 0..num_detections {
            // Get the 84 values for this detection
            let cx: f32;
            let cy: f32;
            let w: f32;
            let h: f32;
            let mut class_scores = [0.0f32; 80];
            
            if is_transposed {
                // Shape (1, 84, 8400): data order is [x0,y0,w0,h0,class0_0,...,class79_0, x1,y1,w1,h1,...]
                cx = data[i];
                cy = data[i + 8400];
                w = data[i + 2 * 8400];
                h = data[i + 3 * 8400];
                for c in 0..80 {
                    class_scores[c] = data[i + (4 + c) * 8400];
                }
            } else {
                // Shape (1, 8400, 84): data order is [x0,y0,w0,h0,class0_0,...,class79_0, x1,y1,w1,h1,...]
                let base = i * 84;
                cx = data[base];
                cy = data[base + 1];
                w = data[base + 2];
                h = data[base + 3];
                for c in 0..80 {
                    class_scores[c] = data[base + 4 + c];
                }
            }
            
            // Find best class score
            let mut best_class_score = 0.0f32;
            let mut best_class = 0usize;
            
            for (c, &score) in class_scores.iter().enumerate() {
                if score > best_class_score {
                    best_class_score = score;
                    best_class = c;
                }
            }
            
            let confidence = best_class_score;
            
            if confidence < self.conf_threshold {
                continue;
            }
            
            // Convert center coords to top-left, then adjust for letterbox
            let x1 = (cx - w / 2.0 - dx_f32) * scale_inv;
            let y1 = (cy - h / 2.0 - dy_f32) * scale_inv;
            let w_scaled = w * scale_inv;
            let h_scaled = h * scale_inv;

            // Clamp to image bounds
            let x_clamped = x1.max(0.0).min(original_w - 1.0);
            let y_clamped = y1.max(0.0).min(original_h - 1.0);
            let w_clamped = w_scaled.max(1.0).min(original_w - x_clamped);
            let h_clamped = h_scaled.max(1.0).min(original_h - y_clamped);

            if w_clamped <= 0.0 || h_clamped <= 0.0 {
                continue;
            }

            bboxes.push(Rect::new(x_clamped as i32, y_clamped as i32, w_clamped as i32, h_clamped as i32));
            confidences.push(confidence);
            class_ids.push(best_class);
            
            // Only print first 10 detections to avoid spam
            if i < 10 {
                println!("[DETECTOR_RAW] Detection #{}: conf={:.3} class={} bbox=({},{},{}x{})",
                    i, confidence, best_class, x_clamped as i32, y_clamped as i32, w_clamped as i32, h_clamped as i32);
            }
        }

        println!("[DETECTOR_FILTERED] Before NMS: {} detections", bboxes.len());

        // ---- Apply NMS ----
        let mut indices = core::Vector::<i32>::new();
        if !bboxes.is_empty() {
            opencv::dnn::nms_boxes(
                &bboxes,
                &confidences,
                self.conf_threshold,
                self.nms_threshold,
                &mut indices,
                1.0,
                0,
            )?;
        }

        println!("[DETECTOR_NMS] After NMS: {} detections", indices.len());

        // ---- Build result ----
        let mut results = Vec::new();
        for i in indices {
            let idx = i as usize;
            results.push(Detection {
                bbox: bboxes.get(idx)?,
                confidence: confidences.get(idx)?,
                class_id: class_ids[idx],
                class_name: "person".to_string(),
            });
        }

        Ok(results)
    }
}
