mod bbox;
mod config;
mod debug;
mod detector;
mod renderer;
mod sort;
mod tracker;
mod video;

use opencv::{highgui, core};
use opencv::prelude::MatTraitConst;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = config::Config::default();

    println!("=== STAGE 1: Frame Validation ===");
    println!("Initializing video source: {}", cfg.video_source);
    let mut video_source = video::VideoSource::new(&cfg.video_source)?;

    // Create debug context
    let dbg = debug::DebugContext::new("debug")?;

    println!("=== STAGE 2: Detector Initialization ===");
    println!("Initializing YOLO Detector from {}", cfg.model_path);
    let mut detector = detector::YoloDetector::new(
        &cfg.model_path,
        cfg.conf_threshold,
        cfg.nms_threshold,
    )?;

    println!("=== STAGE 3: Tracker Initialization ===");
    println!("Initializing SORT Tracker");
    let mut tracker = tracker::Tracker::new(
        cfg.max_age,
        cfg.min_hits,
        cfg.iou_threshold,
    );

    let window_name = "Real-Time Person Tracking";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    println!("\n=== STAGE 4: Main Processing Loop ===");
    let mut frame_count = 0u32;
    let mut total_time = 0.0;
    let mut max_track_id = 0usize;
    let mut frame_detections = Vec::new();
    let mut frame_tracks = Vec::new();

    while let Ok(Some(mut frame)) = video_source.read_frame() {
        let start = Instant::now();
        frame_count += 1;

        // ---- VALIDATION STAGE 1: Frame Info ----
        let frame_size = frame.size()?;
        let width = frame_size.width;
        let height = frame_size.height;
        let channels = frame.channels();

        dbg.log_frame_info(frame_count, width, height, channels);

        // Save first frame for inspection
        if frame_count == 1 {
            dbg.save_frame(&frame, "frame_001.jpg")?;
            println!("[FRAME] Saved initial frame: {}x{} with {} channels", width, height, channels);
        }

        // ---- VALIDATION STAGE 2: Detection ----
        println!("\n--- Frame {} Detection ---", frame_count);
        let detections = detector.detect(&frame)?;
        let detection_count = detections.len();
        
        println!("[DETECTION_COUNT] Frame {}: {} detections", frame_count, detection_count);
        
        if detection_count == 0 && frame_count <= 5 {
            println!("[WARNING] No detections in frame {}! Check detector.", frame_count);
        }

        frame_detections.clear();
        for (idx, det) in detections.iter().enumerate() {
            dbg.log_detection(
                frame_count,
                idx,
                det.bbox.x,
                det.bbox.y,
                det.bbox.width,
                det.bbox.height,
                det.confidence,
                &det.class_name,
            );
            frame_detections.push((det.bbox, det.confidence, det.class_id, det.class_name.clone()));
        }
        dbg.log_detection_summary(frame_count, detection_count, detection_count, detection_count);

        // ---- VALIDATION STAGE 3: Detection Visualization ----
        if frame_count == 1 || frame_count == 2 {
            let mut detection_frame = frame.clone();
            for (bbox, conf, _class_id, class_name) in &frame_detections {
                // Draw detection box in CYAN (without tracking)
                opencv::imgproc::rectangle(
                    &mut detection_frame,
                    *bbox,
                    core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Cyan
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )?;

                let label = format!("{}: {:.2}", class_name, conf);
                opencv::imgproc::put_text(
                    &mut detection_frame,
                    &label,
                    core::Point::new(bbox.x, bbox.y - 5),
                    opencv::imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                    1,
                    opencv::imgproc::LINE_AA,
                    false,
                )?;
            }
            if detection_count > 0 {
                dbg.save_frame(&detection_frame, "detection_overlay.jpg")?;
                println!("[DEBUG] Saved detection visualization with {} boxes", detection_count);
            }
        }

        // ---- VALIDATION STAGE 4: Tracking ----
        println!("\n--- Frame {} Tracking ---", frame_count);
        let tracked_objects = tracker.update(&detections);
        let track_count = tracked_objects.len();

        println!("[TRACK_COUNT] Frame {}: {} active tracks", frame_count, track_count);

        frame_tracks.clear();
        for obj in &tracked_objects {
            dbg.log_track(
                frame_count,
                obj.id,
                obj.bbox.x as i32,
                obj.bbox.y as i32,
                obj.bbox.w as i32,
                obj.bbox.h as i32,
                0, // age would need to be exposed from tracker
                0, // hits would need to be exposed from tracker
            );
            frame_tracks.push(obj.clone());

            if obj.id > max_track_id {
                max_track_id = obj.id;
            }
        }
        dbg.log_track_summary(frame_count, track_count, max_track_id);

        // ---- VALIDATION STAGE 5: Tracking Visualization ----
        if frame_count == 1 || frame_count == 2 {
            let mut tracking_frame = frame.clone();
            for obj in &frame_tracks {
                // Draw tracking box in RED
                opencv::imgproc::rectangle(
                    &mut tracking_frame,
                    obj.bbox.to_rect(),
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )?;

                let label = format!("ID: {} | {} | {:.2}", obj.id, obj.class_name, obj.confidence);
                opencv::imgproc::put_text(
                    &mut tracking_frame,
                    &label,
                    core::Point::new(obj.bbox.x as i32, obj.bbox.y as i32 - 5),
                    opencv::imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                    1,
                    opencv::imgproc::LINE_AA,
                    false,
                )?;
            }
            if track_count > 0 {
                dbg.save_frame(&tracking_frame, "tracking_overlay.jpg")?;
                println!("[DEBUG] Saved tracking visualization with {} tracks", track_count);
            }
        }

        // ---- VALIDATION STAGE 6: Renderer ----
        println!("\n--- Frame {} Rendering ---", frame_count);
        for obj in tracked_objects {
            dbg.log_render(
                frame_count,
                obj.bbox.x as i32,
                obj.bbox.y as i32,
                obj.bbox.w as i32,
                obj.bbox.h as i32,
                obj.id,
            );
            renderer::Renderer::draw_detection(
                &mut frame,
                obj.bbox.to_rect(),
                obj.id,
                &obj.class_name,
                obj.confidence,
            )?;
        }

        // Display footfall count
        renderer::Renderer::draw_footfall_count(&mut frame, max_track_id)?;

        let duration = start.elapsed();
        let fps = 1.0 / duration.as_secs_f32();
        total_time += duration.as_secs_f32();

        // Display FPS
        let fps_text = format!("FPS: {:.1}", fps);
        opencv::imgproc::put_text(
            &mut frame,
            &fps_text,
            core::Point::new(10, 30),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            opencv::imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow(window_name, &frame)?;

        // Summary for first 5 frames
        if frame_count <= 5 {
            println!("[SUMMARY] Frame {}: {} detections -> {} tracks | FPS: {:.1}\n", 
                frame_count, detection_count, track_count, fps);
        }

        let key = highgui::wait_key(1)?;
        if key == 113 || key == 27 {
            break;
        }
    }

    println!("\n=== FINAL REPORT ===");
    if frame_count > 0 {
        println!("Processed frames: {}", frame_count);
        println!("Average FPS: {:.1}", frame_count as f32 / total_time);
        println!("Total unique tracks: {}", max_track_id);
        println!("Debug artifacts saved to: debug/");
    }

    Ok(())
}
