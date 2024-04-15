
const max_u32: u32 = 4294967295u;

@group(0) @binding(10)
var<uniform> u_bound: f32;
@group(0) @binding(11)
var<uniform> v_bound: f32;

/// len is MAX_TRIANGLES * 3
@group(0) @binding(15)
var<storage, read> triangles: array<u32>;

/// len is MAX_TRIANGLES * 3
@group(0) @binding(16)
var<storage, read> halfedges: array<u32>;

/// len is 2 * MAX_VERTICES
@group(0) @binding(17)
var<storage, read> vertices: array<f32>;

/// len is MAX_VERTICES
@group(0) @binding(18)
var<storage, read> height: array<f32>;

/// len is 2 * MAX_VERTICES
@group(0) @binding(19)
var<storage, read> gradients: array<f32>;

/// len is TERRAIN_VERTICES
@group(0) @binding(20)
var<storage, read_write> triangle_indices: array<u32>;

/// len is TERRAIN_VERTICES * 2
@group(0) @binding(21)
var<storage, read> mesh_vertices: array<f32>;

@group(0) @binding(22)
var<storage, read> transform: vec2<f32>;

/// Next halfedge in a triangle.
fn next_halfedge(i: u32) -> u32 {
    if i % 3u == 2u {
        return i - 2u;
    } else {
        return i + 1u;
    }
}

/// Previous halfedge in a triangle.
fn prev_halfedge(i: u32) -> u32 {
    if i % 3u == 0u {
        return i + 2u;
    } else {
        return i - 1u;
    }
}

/// Copied and adapted from
/// https://observablehq.com/@mootari/delaunay-findtriangle
/// Returns the orientation of three points A, B and C:
///   -1 = counterclockwise
///    0 = collinear
///    1 = clockwise
/// More on the topic: http://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf
fn orientation(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    // Determinant of vectors of the line segments AB and BC:
    // [ cx - bx ][ bx - ax ]
    // [ cy - by ][ by - ay ]
    return sign((c.x - b.x) * (b.y - a.y) - (c.y - b.y) * (b.x - a.x));
}

fn find_triangle(p: vec2<f32>, edge: u32) -> u32 {
    if edge == max_u32 {
        return edge;
    }
    // coords is required for delaunator compatibility.
    var current = edge;
    var start = current;
    var n: u32 = 0u;

    // we don't want to go more than 20 times
    // TODO: figure out why having a max of 100 is too much???
    for (var i: i32 = 0; i < 25; i++) {
        let next: u32 = next_halfedge(current);
        let pc: u32 = 2u * triangles[current];
        let pn: u32 = 2u * triangles[next];
//
        let a = vec2<f32>(vertices[pc], vertices[pc + 1u]);
        let b = vec2<f32>(vertices[pn], vertices[pn + 1u]);
//
        let ori: f32 = orientation(a, b, p);
//
        let branch = u32(ori < 0.); // 0 for 1st branch, 1 for 2nd branch

        if start == next && branch == 0u {
            return current;
        }
        current = next * (1u - branch) + current * branch;

        // second branch begin
        if halfedges[current] == max_u32 && branch == 1u {
            return current;
        }
        current = halfedges[current] * branch + current * (1u - branch);
        n += branch;
        let odd = n % 2u;
        current = branch * (odd * next_halfedge(current) + (1u - odd) * prev_halfedge(current)) + (1u - branch) * current;
        start = branch * current + (1u - branch) * start;

//        if ori >= 0. {
//            current = next;
//            if start == current {
//                return current;
//            }
//        } else {
//            if halfedges[current] == max_u32  {
//                return current;
//            }
//            current = halfedges[current];
//            n += 1u;
//            if n % 2u != 0u {
//                current = next_halfedge(current);
//            } else {
//                current = prev_halfedge(current);
//            }
//            start = current;
//        }
    }
    return current;
}


@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // for (var i: u32 = 0u; i < 100u; i++) {
    //     let vertex_idx = invocation_id.x * 100u + i;
    //     let vertex = vec2<f32>(mesh_vertices[vertex_idx*2u] + transform.x, mesh_vertices[vertex_idx*2u + 1u] + transform.y);

    //     var previous = 0u;
    //     if invocation_id.x > 0u {
    //         previous = triangle_indices[vertex_idx - 1u];
    //     }
    //     if previous == max_u32 {
    //         previous = 0u;
    //     }
    //     let found = find_triangle(vertex, previous);
    //     triangle_indices[vertex_idx] = found;
    // }
}


@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let vert_count = arrayLength(&triangle_indices);
    if invocation_id.x < vert_count {
        let vertex_idx = invocation_id.x;
        let vertex = vec2<f32>(
            mesh_vertices[vertex_idx * 2u     ] + transform.x,
            mesh_vertices[vertex_idx * 2u + 1u] + transform.y
        );

        var previous = triangle_indices[vertex_idx];
//        if previous == max_u32 {
//            previous = 0u;
//        }
         let found = find_triangle(vertex, previous);
         triangle_indices[vertex_idx] = found;
    }

//     let vert_count = arrayLength(&mesh_vertices) / 2u;
//     let average_batch_size = u32(ceil(f32(vert_count) / 1024.0f));
//     var batch_size = average_batch_size;
//     let leftover_batch_size = vert_count % 1024u;
//     if invocation_id.x == 1023u && leftover_batch_size > 0u {
//         batch_size = leftover_batch_size;
//     }

//     let batch = invocation_id.x * average_batch_size;
//     for (var i: u32 = 0u; i < batch_size; i++) {
//         let vertex_idx = batch + i;
//         let vertex = vec2<f32>(
//             mesh_vertices[vertex_idx * 2u     ] + transform.x,
//             mesh_vertices[vertex_idx * 2u + 1u] + transform.y
//         );

//         var previous = triangle_indices[vertex_idx];
// //        if previous == max_u32 {
// //            previous = 0u;
// //        }
//         let found = find_triangle(vertex, previous);
//         triangle_indices[vertex_idx] = found;
//     }
}