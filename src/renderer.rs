use opencv::{
    core::{self, Point, Rect, Scalar},
    imgproc,
    prelude::MatTraitConst,
    Result,
};

/// Color configuration: strictly RED (BGR format in OpenCV)
const BBOX_COLOR: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0);
const BBOX_THICKNESS: i32 = 2;

pub struct Renderer;

impl Renderer {
    /// Display footfall count on the frame (top-right corner)
    pub fn draw_footfall_count(
        frame: &mut core::Mat,
        total_count: usize,
    ) -> opencv::Result<()> {
        let label = format!("Footfall: {}", total_count);
        let font_face = imgproc::FONT_HERSHEY_SIMPLEX;
        let font_scale = 1.2;
        let thickness = 2;
        let mut baseline = 0;
        let text_size = imgproc::get_text_size(
            &label,
            font_face,
            font_scale,
            thickness,
            &mut baseline,
        )?;

        // Position in top-right corner
        let x = frame.cols() - text_size.width - 20;
        let y = 40;

        // Draw background
        let bg_rect = Rect::new(
            x - 10,
            y - text_size.height - 10,
            text_size.width + 20,
            text_size.height + baseline + 10,
        );
        imgproc::rectangle(
            frame,
            bg_rect,
            Scalar::new(0.0, 0.0, 0.0, 0.0), // Black background
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )?;

        // Draw text in white
        imgproc::put_text(
            frame,
            &label,
            core::Point::new(x, y),
            font_face,
            font_scale,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            thickness,
            imgproc::LINE_AA,
            false,
        )?;

        Ok(())
    }

    /// Draws a RED bounding box and a label on the given frame.
    /// Label format: `ID: <id> | <class> | <confidence>`
    pub fn draw_detection(
        frame: &mut core::Mat,
        rect: Rect,
        id: usize,
        class_name: &str,
        confidence: f32,
    ) -> Result<()> {
        // Draw the red bounding box
        imgproc::rectangle(
            frame,
            rect,
            BBOX_COLOR,
            BBOX_THICKNESS,
            imgproc::LINE_8,
            0,
        )?;

        // Prepare the label
        let label = format!("ID: {} | {} | {:.2}", id, class_name, confidence);

        // Calculate text size for the background rectangle
        let font_face = imgproc::FONT_HERSHEY_SIMPLEX;
        let font_scale = 0.5;
        let thickness = 1;
        let mut baseline = 0;
        let text_size = imgproc::get_text_size(
            &label,
            font_face,
            font_scale,
            thickness,
            &mut baseline,
        )?;

        // Position label above the bounding box
        let mut label_pos = Point::new(rect.x, rect.y - 5);
        if label_pos.y < text_size.height {
            // If the box is too close to the top, put the label inside the box
            label_pos.y = rect.y + text_size.height + 5;
        }

        // Clamp label position to ensure it stays within reasonable bounds
        label_pos.x = label_pos.x.max(0);
        label_pos.y = label_pos.y.max(text_size.height);

        // Draw a filled background for the text for readability
        let mut bg_rect = Rect::new(
            label_pos.x,
            label_pos.y - text_size.height,
            text_size.width,
            text_size.height + baseline,
        );

        // Clamp background rectangle to prevent it from going off-screen
        if bg_rect.x < 0 {
            bg_rect.width = (bg_rect.width + bg_rect.x).max(0);
            bg_rect.x = 0;
        }
        if bg_rect.y < 0 {
            bg_rect.y = 0;
        }
        imgproc::rectangle(
            frame,
            bg_rect,
            Scalar::new(0.0, 0.0, 0.0, 0.0), // Black background
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )?;

        // Draw the text (White text)
        imgproc::put_text(
            frame,
            &label,
            label_pos,
            font_face,
            font_scale,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            thickness,
            imgproc::LINE_AA,
            false,
        )?;

        Ok(())
    }
}
