use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait},
    Result,
};

pub struct VideoSource {
    capture: VideoCapture,
}

impl VideoSource {
    pub fn new(source: &str) -> Result<Self> {
        let capture = if let Ok(index) = source.parse::<i32>() {
            VideoCapture::new(index, videoio::CAP_ANY)?
        } else {
            VideoCapture::from_file(source, videoio::CAP_ANY)?
        };

        if !VideoCapture::is_opened(&capture)? {
            return Err(opencv::Error::new(
                core::StsError,
                format!("Failed to open video source: {}", source),
            ));
        }

        Ok(Self { capture })
    }

    pub fn read_frame(&mut self) -> Result<Option<core::Mat>> {
        let mut frame = core::Mat::default();
        let success = self.capture.read(&mut frame)?;
        if success && !frame.empty() {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }
}
