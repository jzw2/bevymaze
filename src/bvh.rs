use bevy::utils::{HashMap, HashSet};
use delaunator::{triangulate, Point};
use itertools::assert_equal;
use kiddo::{KdTree, SquaredEuclidean};
use rand::{thread_rng, Rng};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;

type TPoint2d = [f32; 2];

/// top left, bottom right
#[derive(Clone)]
pub struct Aabb2d(TPoint2d, TPoint2d);

impl Aabb2d {
    fn from_indices(vertices: &Vec<TPoint2d>, i1: usize, i2: usize, i3: usize) -> Aabb2d {
        let left = vertices[i1][0].min(vertices[i2][0]).min(vertices[i3][0]);
        let top = vertices[i1][1].min(vertices[i2][1]).min(vertices[i3][1]);

        let right = vertices[i1][0].max(vertices[i2][0]).max(vertices[i3][0]);
        let bottom = vertices[i1][1].max(vertices[i2][1]).max(vertices[i3][1]);

        return Aabb2d([left, top], [right, bottom]);
    }

    fn center(&self) -> TPoint2d {
        let Aabb2d(bb0, bb1) = self;
        let s_x = bb0[0] + bb1[0];
        let s_y = bb0[1] + bb1[1];
        return [s_x / 2., s_y / 2.];
    }

    /// Determines if two BB's intersect.
    /// It does so by creating a larger bounding box.
    /// If both dimensions of the larger box are not strictly greater than the dimensions
    /// of the two boxes combined, the two boxes must intersect
    fn intersects(&self, other: &Aabb2d) -> bool {
        // create the master box
        let bb = self.compose(&other);

        let bb_width = (bb.0[0] - bb.1[0]).abs();
        let bb_height = (bb.0[1] - bb.1[1]).abs();

        let comb_width = (self.1[0] - self.0[0]) + (other.1[0] - other.0[0]);
        let comb_height = (self.1[1] - self.0[1]) + (other.1[1] - other.0[1]);

        return bb_width <= comb_width && bb_height <= comb_height;
    }

    fn compose(&self, other: &Aabb2d) -> Aabb2d {
        let left = self.0[0].min(other.0[0]);
        let top = self.0[1].min(other.0[1]);
        let right = self.1[0].max(other.1[0]);
        let bottom = self.1[0].max(other.1[0]);
        return Aabb2d([left, top], [right, bottom]);
    }
}

/// 3 verts
type Triangle2d = (TPoint2d, TPoint2d, TPoint2d);

#[derive(Clone)]
pub struct Bvh2d {
    pub(crate) triangle: Option<Triangle2d>,
    pub(crate) aabb: Option<Aabb2d>,

    pub(crate) children: Vec<Rc<Bvh2d>>,
}

impl Bvh2d {
    fn build(vertices: &Vec<TPoint2d>, indices: Vec<usize>) -> Bvh2d {
        // convert all the triangles into AABB's first and collect the AABB's into a KD tree for speed
        // the bvh num is an identifier that's guaranteed to be unique (the hash technically isn't, and it scares me)
        let mut bbs = HashMap::<u64, Rc<Bvh2d>>::new();
        let mut bvh_num = 0;
        for i in 0..indices.len() / 3 {
            let i1 = indices[i * 3];
            let i2 = indices[i * 3 + 1];
            let i3 = indices[i * 3 + 2];
            let bb = Aabb2d::from_indices(vertices, i1, i2, i3);
            bbs.insert(
                bvh_num,
                Rc::new(Bvh2d {
                    aabb: Some(bb),
                    triangle: Some((vertices[i1], vertices[i2], vertices[i3])),
                    children: vec![],
                }),
            );
            bvh_num += 1;
        }

        // This helper takes a layer and successively combines multiples of two BVH nodes
        // If there's a leftover node, it simply adds that node to the coalesced output without
        // "bundling" it with anything else
        fn coalesce(bvh_num: &mut u64, bbs: &HashMap<u64, Rc<Bvh2d>>) -> HashMap<u64, Rc<Bvh2d>> {
            let mut tree = KdTree::<f32, 2>::new();
            for (id, bvh) in bbs {
                if let Some(bb) = &bvh.aabb {
                    tree.add(&bb.center(), *id);
                }
            }
            let this_layer = bbs.clone();
            let mut to_coalesce = bbs.clone();
            let mut doubles = HashMap::<u64, Rc<Bvh2d>>::new();
            // add all our things to the doubles and remove them from the originals
            while to_coalesce.len() > 0 {
                // println!("coal len {}", to_coalesce.len());
                if to_coalesce.len() == 1 {
                    // leftover, add it and return
                    let (id1, bvh1) = to_coalesce.iter().next().unwrap();
                    doubles.insert(*id1, bvh1.clone());
                    return doubles;
                }

                let mut used = &mut (0, 0);

                {
                    let (id1, bvh1) = to_coalesce.iter().next().unwrap();
                    let bb1 = &bvh1.aabb.clone().unwrap();
                    // remove it from the tree so we don't find ourselves
                    tree.remove(&bb1.center(), *id1);
                    // the nearest one will be our partner in the upper BB
                    let nearest = tree.nearest_one::<SquaredEuclidean>(&bb1.center());
                    let bvh2 = bbs.get(&nearest.item).unwrap();
                    let bb2 = &bvh2.aabb.clone().unwrap();
                    // remove this second one now so it can't be paired with anything
                    tree.remove(&bb2.center(), nearest.item);
                    let enclosing_bb = bb1.compose(bb2);
                    let mut enclosing_bvh = Bvh2d {
                        triangle: None,
                        aabb: Some(enclosing_bb.clone()),
                        children: vec![],
                    };
                    // finally find all the children that intersect it
                    // here's where we add the two children that we made the BB out of in the first place,
                    // as well as any other BB's that happen to intersect
                    for nearest_n in tree.nearest_n::<SquaredEuclidean>(&bb1.center(), 6) {
                        let bvh = this_layer.get(&nearest_n.item).unwrap();
                        if let Some(bb) = &bvh.aabb {
                            if enclosing_bb.intersects(&bb) {
                                enclosing_bvh.children.push(Rc::clone(bvh));
                            }
                        }
                    }
                    // bbs.insert(bvh_num, enclosing_bvh);
                    doubles.insert(*bvh_num, Rc::new(enclosing_bvh));
                    *bvh_num += 1;
                    (*used).0 = *id1;
                    (*used).1 = nearest.item;
                    // used = (*id1, nearest.item);
                }
                // finally we remove these from the ones we needed to coalesce
                to_coalesce.remove(&used.0);
                to_coalesce.remove(&used.1);
            }
            return doubles;
        }

        // coalesce until there's only one node left
        while bbs.len() > 1 {
            println!("bb len {}", bbs.len());
            bbs = coalesce(&mut bvh_num, &bbs);
        }

        // there should be one left at this point
        let next = bbs.iter().next().unwrap();
        let next = next.1;
        return next.deref().clone();
    }

    fn intersects_bb(&self, point: TPoint2d) -> bool {
        if let Some(bb) = &self.aabb {
            let top_left = bb.0;
            let bottom_right = bb.1;
            return (top_left[0] <= point[0] && point[0] <= bottom_right[0])
                && (top_left[1] <= point[1] && point[1] <= bottom_right[1]);
        }
        return false;
    }

    /// Adapted from https://stackoverflow.com/a/14382692/3210986
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
        for bvh in &self.children {
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
        for bvh in &self.children {
            let mut arr = bvh.to_array();
            buffer.push(arr.len() as u32);
            buffer.append(&mut arr);
        }
        return buffer;
    }
}

#[test]
fn build_find_test() {
    // generate three points
    let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let bvh = Bvh2d::build(&points, vec![0, 1, 2]);
    let tri = bvh.get_tri([0.25, 0.25]);
    if let Some(triangle) = tri {
        println!(
            "Triangle found: ({}, {}) ({}, {}) ({}, {})",
            triangle.0[0],
            triangle.0[1],
            triangle.1[0],
            triangle.1[1],
            triangle.2[0],
            triangle.2[1]
        );
    } else {
        panic!("No triangle found D:")
    }

    let tri = bvh.get_tri([-0.25, -0.25]);
    if let Some(triangle) = tri {
        panic!(
            "Triangle found: ({}, {}) ({}, {}) ({}, {})",
            triangle.0[0],
            triangle.0[1],
            triangle.1[0],
            triangle.1[1],
            triangle.2[0],
            triangle.2[1]
        );
    } else {
        println!("No triangle found :D")
    }
}

#[test]
fn two_layers_test() {
    let mut points = vec![];
    let mut delaunay_points: Vec<Point> = vec![];
    for _ in 0..4 {
        let p = [
            thread_rng().gen_range(-5.0..5.0),
            thread_rng().gen_range(-5.0..5.0),
        ];
        points.push([p[0] as f32, p[1] as f32]);
        delaunay_points.push(Point { x: p[0], y: p[1] });
    }
    let triangulation = triangulate(&delaunay_points);
    let bvh = Bvh2d::build(&points, triangulation.triangles);
    // make sure there are 2 layers
    let child = &bvh.children[0];
    assert_eq!(child.children.len(), 0);
    // make sure there's a triangle here
    child.triangle.unwrap();
}

#[test]
fn many_layers_test() {
    let mut points = vec![];
    let mut delaunay_points: Vec<Point> = vec![];
    println!("Generating points");
    for _ in 0..100000 {
        let p = [
            thread_rng().gen_range(-5.0..5.0),
            thread_rng().gen_range(-5.0..5.0),
        ];
        points.push([p[0] as f32, p[1] as f32]);
        delaunay_points.push(Point { x: p[0], y: p[1] });
    }
    println!("Triangulating");
    let triangulation = triangulate(&delaunay_points);
    println!("Building BVH");
    let bvh = Bvh2d::build(&points, triangulation.triangles);
    // make sure there are at least 2 layers
    println!("Checking BVH");
    let child = &bvh.children[0];

    println!("Walking");
    // make sure we can walk the bvh
    let tri = bvh.get_tri([0.25, 0.25]);
    if let Some(triangle) = tri {
        println!(
            "Triangle found: ({}, {}) ({}, {}) ({}, {})",
            triangle.0[0],
            triangle.0[1],
            triangle.1[0],
            triangle.1[1],
            triangle.2[0],
            triangle.2[1]
        );
    } else {
        panic!("No triangle found D:")
    }
}
