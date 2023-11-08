use server::terrain_data::DATUM_COUNT;

#[cfg(test)]

// Note this useful idiom: importing names from outer (for mod tests) scope.

#[test]
fn test_add() {
    let buffer = vec![0u16; DATUM_COUNT];
    generate_data(buffer, (0, 0));
    //assert_eq!(add(1, 2), 3);
}
