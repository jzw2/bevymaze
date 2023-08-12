use image::{ImageBuffer, Rgb, Rgba, RgbaImage, RgbImage};
use itertools::iproduct;
use crate::terrain_gen::{generate_tile, TerrainGenerator};
use crate::util::lin_map;

mod terrain_gen;
mod util;

fn main() {
    let terrain_gen = TerrainGenerator::new();
    let tile1 = generate_tile(&terrain_gen, 0, 0);
    let mut image = RgbImage::from_pixel(1024, 1024, Rgb([255, 255, 255]));
    for (px, py) in iproduct!(0..1024, 0..1024) {
        let pixel_val = f32::from_bits(tile1[px][py]);
        let pixel_val = lin_map(0., 128. + 90. + 4., 0., 255., pixel_val as f64) as u8;
        image.put_pixel(px as u32, py as u32, Rgb([pixel_val, pixel_val, pixel_val]));
    }
    image.save("output.png").unwrap();
}
