use image::{Rgb, RgbImage};
use std::cmp::{max, min};

pub struct Circle {
    pub(crate) center: (f64, f64),
    pub(crate) radius: f64,
    pub(crate) line_width: f64,
    pub(crate) color: Rgb<u8>,
}

pub struct Segment {
    pub(crate) p1: (f64, f64),
    pub(crate) p2: (f64, f64),
    pub(crate) line_width: f64,
    pub(crate) color: Rgb<u8>,
}

#[derive(Clone, Copy)]
pub struct AxisTransform {
    pub(crate) offset: (f64, f64),
    pub(crate) scale: (f64, f64),
}

pub fn draw_circle(image: &mut RgbImage, circle: Circle, transform: AxisTransform) {
    for px in 0..image.width() {
        for py in 0..image.height() {
            let x = (px as f64 - transform.offset.0) / transform.scale.0;
            let y = (py as f64 - transform.offset.1) / transform.scale.1;

            let x = x - circle.center.0;
            let y = y - circle.center.1;

            let center_dist = (x * x + y * y).sqrt();
            let circle_dist = (circle.radius - center_dist).abs();
            if circle_dist <= circle.line_width / 2.0 {
                image.put_pixel(px, py, circle.color);
            }
        }
    }
}

pub fn draw_segment(image: &mut RgbImage, segment: Segment, transform: AxisTransform) {
    for px in 0..image.width() {
        for py in 0..image.height() {
            let x = (px as f64 - transform.offset.0) / transform.scale.0;
            let y = (py as f64 - transform.offset.1) / transform.scale.1;

            // get the projection of p onto line p1 p2
            let v = (segment.p2.0 - segment.p1.0, segment.p2.1 - segment.p1.1);
            let p_off = (x - segment.p1.0, y - segment.p1.1);
            let t = (v.0 * p_off.0 + v.1 * p_off.1) / (v.0 * v.0 + v.1 * v.1);
            // clamp the projection
            let t = 0.0f64.max(1.0f64.min(t));
            // now get the dist
            let seg_p = (x - (t * v.0 + segment.p1.0), y - (t * v.1 + segment.p1.1));
            let seg_dist = (seg_p.0 * seg_p.0 + seg_p.1 * seg_p.1).sqrt();
            if seg_dist <= segment.line_width / 2.0 {
                image.put_pixel(px, py, segment.color);
            }
        }
    }
}
