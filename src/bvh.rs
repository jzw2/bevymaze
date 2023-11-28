
type TPoint2d = (f32, f32);

/// top left, bottom right
type Aabb2d = (TPoint2d, TPoint2d);

/// 3 verts
type Triangle2d = (TPoint2d, TPoint2d, TPoint2d);

struct Bvh2d {
    triangle: Option<Triangle2d>,
    aabb: Option<Aabb2d>,

    left: Option<&Bvh2d>,
    right: Option<&Bvh2d>
}

impl Bvh2d {
    fn intersects_bb(&self, point: TPoint2d) -> bool {
        if let Some(bb) = self.aabb {
            let top_left = bb.0;
            let bottom_right = bb.1;
            return (top_left.0 < point.0 && point.0 < bottom_right.0) && (top_left.1 < point.1 && point.1 < bottom_right.1);
        }
        return false;
    }

    fn intersects_tri(&self, point: TPoint2d) -> bool {
        if let Some(tri) = self.triangle {
            let p1 = tri.0;
            let p2 = tri.1;
            let p3 = tri.2;
            let area = 0.5 *(-p1.1*p2.0 + p0.1*(-p1.0 + p2.0) + p0.0*(p1.1 - p2.1) + p1.0*p2.1);
            let s = 1./(2.*area)*(p0.1*p2.0 - p0.0*p2.1 + (p2.1 - p0.1)*point.0 + (p0.0 - p2.0)*point.1);
            let t = 1./(2.*area)*(p0.0*p1.1 - p0.1*p1.0 + (p0.1 - p1.1)*p0.0 + (p1.0 - p0.0)*point.1);
            return s >= 0.0 && t >= 0 && 1.-s-t >= 0.0;
        }
        return false;
    }

    fn intersects(&self, point: TPoint2d) -> bool {
        return intersects_tri(point) || intersects_bb(point);
    }

    fn get_tri(&self, point: TPoint2d) -> Option<Triangle2d> {
        if self.intersects_tri(point) {
            return self.triangle;
        }
        let mut tri: Option<Triangle2d> = None;
        if let Some(left) = self.left && left.intersects_bb(point) {
            tri = left.get_tri(point);
        }
        if let Some(right) = self.right && right.intersects_bb(point) {
            tri = right.get_tri(point);
        }
        return tri;
    }
    
    fn flatten(&self) -> Vec<u32> {
        let buffer: Vec<u32> = vec![];
        // write the aabb
        if let bb = Some(self.aabb) {
            // flag indicating that it's a bounding box
            buffer.push(f32::NAN.to_bits());
            buffer.push(f32::NAN.to_bits());
            buffer.push(bb.0.0.to_bits());
            buffer.push(bb.0.1.to_bits());
            buffer.push(bb.1.0.to_bits());
            buffer.push(bb.1.1.to_bits());
        } else if let tri = Some(self.triangle) {
            buffer.push(tri.0.0.to_bits());
            buffer.push(tri.0.1.to_bits());
            buffer.push(tri.1.0.to_bits());
            buffer.push(tri.1.1.to_bits());
            buffer.push(tri.2.0.to_bits());
            buffer.push(tri.2.1.to_bits());
        }
        // now write the children
        let left = self.left.flatten(buffer);
        buffer.push(left.len() as u32);
        buffer.append(left);
        buffer.append(self.right.flatten(buffer));
        return buffer;
    }
}

fn aabb_from_indices(i1: usize, i2: usize, i3: usize) -> 

fn build_bvh(vertices: Vec<TPoint2d>, indices: Vec<usize>) -> Bvh2d {
    const bvh_col = 0.0
    // first gather all the triangles
    for let i in 0..indices.len() {
        
    }
}
