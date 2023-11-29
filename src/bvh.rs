use bevy::utils::{HashMap, HashSet};
use kiddo::KdTree;


type TPoint2d = [f32; 2];

/// top left, bottom right
#[derive(Eq, Hash, PartialEq)]
struct  Aabb2d(TPoint2d, TPoint2d);

impl Aabb2d {
    fn center(&self) -> TPoint2d {
        let Aabb2d(bb0, bb1) = self;
        let s_x = bb0[0] + bb1[0];
        let s_y = bb0[1] + bb1[1];
        return [s_x/2., s_y/2.];
    }
}

/// 3 verts
type Triangle2d = (TPoint2d, TPoint2d, TPoint2d);

struct Bvh2d {
    triangle: Option<Triangle2d>,
    aabb: Option<Aabb2d>,

    children: Vec<Bvh2d>,
}

impl Bvh2d {
    fn intersects_bb(&self, point: TPoint2d) -> bool {
        if let Some(bb) = &self.aabb {
            let top_left = bb.0;
            let bottom_right = bb.1;
            return (top_left[0] <= point[0] && point[0] <= bottom_right[0])
                && (top_left[1] <= point[1] && point[1] <= bottom_right[1]);
        }
        return false;
    }

    fn intersects_tri(&self, point: TPoint2d) -> bool {
        if let Some(tri) = self.triangle {
            let p0 = tri.0;
            let p1 = tri.1;
            let p2 = tri.2;
            let area = 0.5
                * (-p1[1] * p2[0]
                    + p0[1] * (-p1[0] + p2[0])
                    + p0[0] * (p1[1] - p2[1])
                    + p1[0] * p2[1]);
            let s = 1. / (2. * area)
                * (p0[1] * p2[0] - p0[0] * p2[1]
                    + (p2[1] - p0[1]) * point[0]
                    + (p0[0] - p2[0]) * point[1]);
            let t = 1. / (2. * area)
                * (p0[0] * p1[1] - p0[1] * p1[0]
                    + (p0[1] - p1[1]) * p0[0]
                    + (p1[0] - p0[0]) * point[1]);
            return s >= 0. && t >= 0. && 1. - s - t >= 0.;
        }
        return false;
    }

    fn get_tri(&self, point: TPoint2d) -> Option<Triangle2d> {
        if self.intersects_tri(point) {
            return self.triangle;
        }
        let mut tri: Option<Triangle2d> = None;
        for bvh in self.children {
            if bvh.intersects_bb(point) {
                return bvh.get_tri(point);
            }
        }
        return tri;
    }

    fn to_array(&self) -> Vec<u32> {
        let mut buffer: Vec<u32> = vec![];
        // write the aabb
        if let Some(bb) = &self.aabb {
            // flag indicating that it's a bounding box
            buffer.push(f32::NAN.to_bits());
            buffer.push(f32::NAN.to_bits());
            buffer.push(bb.0[0].to_bits());
            buffer.push(bb.0[1].to_bits());
            buffer.push(bb.1[0].to_bits());
            buffer.push(bb.1[1].to_bits());
        } else if let Some(tri) = self.triangle {
            buffer.push(tri.0[0].to_bits());
            buffer.push(tri.0[1].to_bits());
            buffer.push(tri.1[0].to_bits());
            buffer.push(tri.1[1].to_bits());
            buffer.push(tri.2[0].to_bits());
            buffer.push(tri.2[1].to_bits());
        }

        // now write the children
        buffer.push(self.children.len() as u32);
        for bvh in self.children {
            let mut arr = bvh.to_array();
            buffer.push(arr.len() as u32);
            buffer.append(&mut arr);
        }
        return buffer;
    }
}

fn aabb_from_indices(vertices: &Vec<TPoint2d>, i1: usize, i2: usize, i3: usize) -> Aabb2d {
    let left = vertices[i1][0].min(vertices[i2][0]).min(vertices[i3][0]);
    let top = vertices[i1][1].min(vertices[i2][1]).min(vertices[i3][1]);

    let right = vertices[i1][0].max(vertices[i2][0]).max(vertices[i3][0]);
    let bottom = vertices[i1][1].max(vertices[i2][1]).max(vertices[i3][1]);

    return Aabb2d([left, top], [right, bottom]);
}

fn build_bvh(vertices: &Vec<TPoint2d>, indices: Vec<usize>) -> Bvh2d {
    // convert all the triangles into AABB's first and collect the AABB's into a KD tree for speed
    // the bvh num is an identifier that's guaranteed to be unique (the hash technically isn't, and it scares me)
    let mut bbs = HashMap::<u64, Aabb2d>::new();
    let mut tree = KdTree::<f32, 2>::new();
    let mut bvh_num = 0;
    for i in 0..indices.len() / 3 {
        let Aabb2d(bb0, bb1) = aabb_from_indices(vertices, i * 3, i * 3 + 1, i * 3 + 2);
        bbs.insert(bvh_num, Aabb2d(bb0, bb1));
        let s_x = bb0[0] + bb1[0];
        let s_y = bb0[1] + bb1[1];
        tree.add(&[s_x/2., s_y/2.], bvh_num);
        bvh_num += 1;
    }
    // grab bbs that are close and combine them
    let Some(bb1) = bbs.iter().next();
    let bb2 = tree.nearest_one(bb1);

    return Bvh2d {};
}
