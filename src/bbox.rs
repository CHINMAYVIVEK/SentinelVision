use opencv::core::Rect;

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn from_rect(rect: Rect) -> Self {
        Self {
            x: rect.x as f32,
            y: rect.y as f32,
            w: rect.width as f32,
            h: rect.height as f32,
        }
    }

    pub fn to_rect(&self) -> Rect {
        Rect::new(self.x as i32, self.y as i32, self.w as i32, self.h as i32)
    }

    /// Convert to [cx, cy, s, r] format for SORT
    /// cx, cy: center x, y
    /// s: scale (area)
    /// r: aspect ratio (w/h)
    pub fn to_z_vector(&self) -> nalgebra::DVector<f32> {
        let cx = self.x + self.w / 2.0;
        let cy = self.y + self.h / 2.0;
        let s = self.w * self.h;
        // avoid division by zero
        let r = if self.h > 0.0 { self.w / self.h } else { 0.0 };
        nalgebra::DVector::from_vec(vec![cx, cy, s, r])
    }

    /// Convert back from [cx, cy, s, r]
    pub fn from_z_vector(z: &nalgebra::DVector<f32>) -> Self {
        let cx = z[0];
        let cy = z[1];
        let s = z[2].max(0.001); // avoid negative scale
        let r = z[3].max(0.001); // avoid negative ratio

        let w = (s * r).sqrt();
        let h = s / w;
        let x = cx - w / 2.0;
        let y = cy - h / 2.0;

        Self { x, y, w, h }
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let x_a = self.x.max(other.x);
        let y_a = self.y.max(other.y);
        let x_b = (self.x + self.w).min(other.x + other.w);
        let y_b = (self.y + self.h).min(other.y + other.h);

        let inter_area = (x_b - x_a).max(0.0) * (y_b - y_a).max(0.0);
        let union_area = self.area() + other.area() - inter_area;

        if union_area == 0.0 {
            return 0.0;
        }

        inter_area / union_area
    }
}
