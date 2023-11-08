use server::terrain_data::{compressed_height, uncompressed_height, TerrainTile, DATUM_COUNT};
use server::terrain_gen::{generate_data, TerrainGenerator, TILE_SIZE};

#[cfg(test)]
// Note this useful idiom: importing names from outer (for mod tests) scope.
#[test]
fn test_compress_decompress_tile_data() {
    let mut buffer = vec![0u16; DATUM_COUNT];
    generate_data(&mut buffer, (0, 0));
    let tile = TerrainTile::from(buffer.clone());
    let terrain_gen = TerrainGenerator::new();
    assert_eq!(
        buffer[0],
        compressed_height(terrain_gen.get_height_for(0., 0.,))
    );
    assert_eq!(tile.get_height(0., 0.), uncompressed_height(buffer[0]));
    assert_eq!(
        buffer[DATUM_COUNT - 1],
        compressed_height(terrain_gen.get_height_for(
            TILE_SIZE - 1. / DATUM_COUNT as f64,
            TILE_SIZE - 1. / DATUM_COUNT as f64,
        ))
    );
    assert_eq!(tile.get_height(TILE_SIZE - 1. / DATUM_COUNT as f64, TILE_SIZE - 1. / DATUM_COUNT as f64,), uncompressed_height(buffer[DATUM_COUNT-1]));
}
