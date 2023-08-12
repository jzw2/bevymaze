pub type SimpleVertex = [f32; 3];
pub type SimpleVertices = Vec<SimpleVertex>;
pub type CompleteVertices = Vec<(SimpleVertex, [f32; 3], [f32; 2])>;
pub type Indices = Vec<u32>;

// get the indices of the face in clockwise order
pub fn tri_cw_indices(cur_idx_set: u32) -> Indices {
    let i = cur_idx_set * 3;
    return vec![i, i + 1, i + 2];
}

// get the indices of the face in counterclockwise order
pub fn tri_cc_indices(cur_idx_set: u32) -> Indices {
    let i = cur_idx_set * 3;
    return vec![i + 2, i + 1, i];
}

// get the indices of the face in clockwise order
// manually provide the offset
pub fn tri_cw_indices_off(off: u32) -> Indices {
    let i = off;
    return vec![i, i + 1, i + 2];
}

// get the indices of the face in counterclockwise order
// manually provide the offset
pub fn tri_cc_indices_off(off: u32) -> Indices {
    let i = off;
    return vec![i + 2, i + 1, i];
}

// get the indices of the face in clockwise order
pub fn quad_cw_indices(cur_idx_set: u32) -> Indices {
    let i = cur_idx_set * 4;
    return vec![i, i + 1, i + 2, i + 3, i + 2, i + 1];
}

// get the indices of the face in counterclockwise order
pub fn quad_cc_indices(cur_idx_set: u32) -> Indices {
    let i = cur_idx_set * 4;
    return vec![i + 2, i + 1, i, i + 1, i + 2, i + 3];
}

// get the indices of the face in clockwise order
// manually provide the offset
pub fn quad_cw_indices_off(off: u32) -> Indices {
    let i = off;
    return vec![i, i + 1, i + 2, i + 3, i + 2, i + 1];
}

// get the indices of the face in counterclockwise order
// manually provide the offset
pub fn quad_cc_indices_off(off: u32) -> Indices {
    let i = off;
    return vec![i + 2, i + 1, i, i + 1, i + 2, i + 3];
}
