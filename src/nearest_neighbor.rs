use crate::shaders::MAX_VERTICES;
use crate::terrain_render::TERRAIN_VERTICES;
use bevy::math::{DVec2, Vec2};
use bevy::prelude::Resource;
use bevy::reflect::List;
use bevy::utils::HashSet;
use delaunator::{next_halfedge, prev_halfedge, triangulate, Point, Triangulation};
use itertools::Itertools;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ordered_float::OrderedFloat;
use server::connection::TerrainDataPoint;
use std::collections::HashMap;

#[derive(Clone)]
pub struct TerrainDataLocator {
    pub last_idx: usize,
    corners: [[f64; 2]; 4],
    data: Vec<[f64; 2]>,
    triangulation: Triangulation,
}

struct NearestNeighbor<'point> {
    /// The actual point
    point: &'point [f64; 2],
    /// The distance to this nearest neighbor
    distance: f64,
    /// The index in the triangulation
    idx: usize,
    /// The index in the data (vertex) array
    data_idx: usize,
}

fn distance_squared(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let px = p1[0] - p2[0];
    let py = p1[1] - p2[1];
    return px * px + py * py;
}

impl TerrainDataLocator {
    fn points_from(data: &Vec<[f64; 2]>, corners: &[[f64; 2]; 4]) -> Vec<Point> {
        let mut points: Vec<Point> = data
            .into_iter()
            .filter_map(|[x, y]| {
                return if x.is_finite() && y.is_finite() {
                    Some(Point { x: *x, y: *y })
                } else {
                    None
                };
            })
            .collect();
        let mut corner_points: Vec<Point> = corners.into_iter().map(|d| Point::from(*d)).collect();
        points.append(&mut corner_points);

        return points;
    }

    pub fn new(data: &Vec<[f64; 2]>) -> Self {
        let corners = [
            [-50_000_000., -50_000_000.], // the radius of the earth is around 40 mil, so go way above that
            [50_000_000., -50_000_000.],
            [-50_000_000., 50_000_000.],
            [50_000_000., 50_000_000.],
        ];
        return TerrainDataLocator {
            last_idx: 0,
            data: data.clone(),
            triangulation: triangulate(&Self::points_from(&data, &corners)),
            corners,
        };
    }

    #[inline]
    fn nearest_around_vertex(
        &self,
        target: &[f64; 2],
        best_dist_sqr: f64,
        mut fulcrum: usize,
    ) -> Option<(f64, usize)> {
        let mut next_spoke = fulcrum;

        if fulcrum == usize::MAX {
            return None;
        }

        let starting_spoke = prev_halfedge(fulcrum);

        loop {
            next_spoke = prev_halfedge(fulcrum);
            let cur_dist_sqr =
                distance_squared(&self.data[self.triangulation.triangles[next_spoke]], target);

            if best_dist_sqr > cur_dist_sqr {
                return Some((cur_dist_sqr, next_spoke));
            }

            // update the fulcrum. It's going to be the opposite half edge of the current spoke
            fulcrum = self.triangulation.halfedges[next_spoke];

            if prev_halfedge(fulcrum) != starting_spoke && fulcrum != usize::MAX {
                return None;
            }
        }
    }

    pub fn nearest(&self, target: &[f64; 2], edge: usize) -> NearestNeighbor {
        // we start at a random vertex
        let mut cur_fulcrum = edge;
        let starting_dist_sqr = distance_squared(
            &self.data[self.triangulation.triangles[cur_fulcrum]],
            target,
        );
        let mut distance_squared = starting_dist_sqr;

        loop {
            if let Some((cur_dist_sqr, closer_spoke)) =
                self.nearest_around_vertex(target, distance_squared, cur_fulcrum)
            {
                distance_squared = cur_dist_sqr;
                cur_fulcrum = closer_spoke;
            } else {
                break;
            }
        }

        return NearestNeighbor {
            point: &self.data[self.triangulation.triangles[cur_fulcrum]],
            distance: distance_squared.sqrt(),
            idx: cur_fulcrum,
            data_idx: self.triangulation.triangles[cur_fulcrum],
        };
    }

    /// Remove the data at idx
    /// This does not remove it per se, but changes the index
    /// NOTE: this DOES NOT update the triangulation
    pub fn remove(&mut self, idx: usize) {
        self.data[idx] = [f64::INFINITY, f64::INFINITY];
    }

    pub fn replace(&mut self, idx: usize, point: &[f64; 2]) {
        self.data[idx] = *point;
    }

    pub fn update(&mut self) {
        self.triangulation = triangulate(&Self::points_from(&self.data, &self.corners));
    }
}

#[derive(Clone)]
pub struct TerrainDataHolder {
    /// The actual data we have on hand
    /// The amount of data is length 2N. The last N elements are dedicated to elements
    /// that have we have not fetched yet
    pub data: Vec<TerrainDataPoint>,
    /// The locator lets us find which point in the terrain data we are closest to quickly
    /// This also includes unresolved data
    locator: TerrainDataLocator,
    /// This is the list of eviction candidates from best to worst
    /// It is sorted by the candidates' ratio of actual distance to ideal distance
    eviction_priority: Vec<(usize, f32)>,
}

#[derive(Clone)]
pub struct TerrainMeshHolder {
    /// An unordered list of vertices in the mesh
    pub terrain_mesh_verts: Vec<[f32; 2]>,
    /// The reverse locator lets us find which point in the terrain MESH we are closest to quickly
    reverse_locator: TerrainDataLocator,
    /// Center of the terrain mesh
    pub center: Vec2,
    /// The radius of acceptance for each vertex in terrain_mesh_verts
    check_radii: Vec<f32>,
}

#[derive(Resource, Clone)]
pub struct TerrainDataMap {
    initialized: bool,
    pub resolved_data: TerrainDataHolder,
    unresolved_data: TerrainDataHolder,
    pub mesh: TerrainMeshHolder,
}

/// How much the nodes are rounded by
const ACCURACY: f32 = 0.25;

const DATA_TOLERANCE: f64 = 0.5;

impl TerrainDataHolder {
    /// Partially sort the priorities by the first n highest
    pub fn select_highest_priorities(&mut self, mesh: &TerrainMeshHolder, n_highest: usize) {
        let mut last_idx = 0;
        for (i, data) in (&self.data).into_iter().enumerate() {
            if data.coordinates[0].is_infinite() || data.coordinates[1].is_infinite() {
                self.eviction_priority[i] = (i, f32::INFINITY);
            } else {
                // correct for coordinate being in world space, we want it relative to the mesh
                let nearest = mesh.reverse_locator.nearest(
                    &[
                        (data.coordinates[0] - mesh.center.x) as f64,
                        (data.coordinates[1] - mesh.center.y) as f64,
                    ],
                    last_idx,
                );
                let required_dist = mesh.get_check_radius(nearest.data_idx);
                self.eviction_priority[i] = (i, nearest.distance as f32 / required_dist);
            }
        }
        self.eviction_priority
            .select_nth_unstable_by(n_highest - 1, |a, b| b.1.total_cmp(&a.1));
    }
}

impl TerrainMeshHolder {
    /// Get the radius of acceptable data for a particular vertex in the terrain mesh
    pub fn get_check_radius(&self, vertex_idx: usize) -> f32 {
        return self.check_radii[vertex_idx];
    }
}

impl TerrainDataMap {
    pub fn new(terrain_lattice: &Vec<DVec2>) -> Self {
        return TerrainDataMap::new_with_capacity(terrain_lattice, MAX_VERTICES);
    }

    pub fn new_with_capacity(terrain_lattice: &Vec<DVec2>, capacity: usize) -> Self {
        let lattice_data: Vec<[f32; 2]> = terrain_lattice
            .into_iter()
            .map(|e| e.as_vec2().to_array())
            .collect();
        let holder = TerrainDataHolder {
            data: vec![
                TerrainDataPoint {
                    coordinates: [f32::INFINITY, f32::INFINITY],
                    height: 0.0,
                    idx: 0,
                    gradient: [0., 0.]
                };
                capacity
            ],
            locator: TerrainDataLocator::new(&vec![[f64::INFINITY; 2]; capacity]),
            eviction_priority: Vec::from_iter(
                vec![f32::INFINITY; capacity].into_iter().enumerate(),
            ),
        };
        let check_radii = terrain_lattice
            .into_iter()
            .map(|e| (DATA_TOLERANCE * e.length()) as f32)
            .collect();

        let reverse_locator = TerrainDataLocator::new(
            &lattice_data
                .clone()
                .into_iter()
                .map(|[x, y]| [x as f64, y as f64])
                .collect(),
        );
        return TerrainDataMap {
            initialized: false,
            resolved_data: holder.clone(),
            unresolved_data: holder,
            mesh: TerrainMeshHolder {
                reverse_locator,
                terrain_mesh_verts: lattice_data,
                center: Vec2::ZERO,
                check_radii,
            },
        };
    }

    /// Mark these vertices as requested data
    /// We potentially evict other requested data before it's been filled in
    pub fn mark_requested(&mut self, data: &mut Vec<TerrainDataPoint>) {
        // add it to the locator to mark it as part of the useful data
        self.unresolved_data
            .select_highest_priorities(&self.mesh, data.len());
        for (i, datum) in data.into_iter().enumerate() {
            // evict any old unresolved data
            let (evicting, _) = self.unresolved_data.eviction_priority[i];

            self.unresolved_data
                .locator
                .replace(evicting, &datum.coordinates.map(|e| e as f64));
            datum.idx = evicting;
            self.unresolved_data.data[evicting] = datum.clone();
        }

        self.unresolved_data.locator.update();
    }

    pub fn fill_in_data(&mut self, data: &Vec<TerrainDataPoint>) -> bool {
        // first check the requested data and make sure this point still exists
        let data = data
            .iter()
            .filter(|new_data| {
                let old_data = &mut self.unresolved_data.data[new_data.idx];
                return old_data.coordinates[0] == new_data.coordinates[0]
                    && old_data.coordinates[1] == new_data.coordinates[1];
            })
            .collect_vec();
        // prioritize for replacement
        self.resolved_data
            .select_highest_priorities(&self.mesh, data.len());
        let empty = data.is_empty();
        // finally replace
        for (i, new_data) in data.into_iter().enumerate() {
            // first check the requested data and make sure this point still exists
            let mut old_unresolved = &mut self.unresolved_data.data[new_data.idx];
            // it does match, so remove this unresolved data
            self.unresolved_data.locator.remove(old_unresolved.idx);
            old_unresolved.coordinates = [f32::INFINITY, f32::INFINITY];

            // now replace the resolved data
            let (evicting, _) = self.resolved_data.eviction_priority[i];

            let old_resolved = &mut self.resolved_data.data[evicting];
            old_resolved.coordinates = new_data.coordinates;
            old_resolved.height = new_data.height;
            old_resolved.gradient = new_data.gradient;
            old_resolved.idx = evicting;
            self.resolved_data.locator.replace(
                old_resolved.idx,
                &old_resolved.coordinates.map(|e| e as f64),
            );
        }

        self.unresolved_data.locator.update();
        self.resolved_data.locator.update();

        return !empty;
    }

    /// Checks whether the terrain data is still acceptable to interpolate the specified point
    /// Data further from the camera has a higher tolerance for being off
    pub fn vert_to_fetch(&mut self, vertex_idx: usize) -> Option<[f32; 2]> {
        let mut vertex = self.mesh.terrain_mesh_verts[vertex_idx];
        vertex[0] += self.mesh.center.x;
        vertex[1] += self.mesh.center.y;
        let vtx = vertex.map(|e| e as f64);
        // we check if the nearest guy is in the acceptable range
        let nearest = self.resolved_data.locator.nearest(&vtx, self.resolved_data.locator.last_idx);
        let unresolved_nearest = self.unresolved_data.locator.nearest(&vtx, self.unresolved_data.locator.last_idx);
        let mut min = f32::INFINITY;
        min = min
            .min(nearest.distance as f32)
            .min(unresolved_nearest.distance as f32);
        if min > self.mesh.get_check_radius(vertex_idx) {
            return Some(vertex);
        }
        self.resolved_data.locator.last_idx = nearest.idx;
        self.unresolved_data.locator.last_idx = unresolved_nearest.idx;
        
        return None;
    }
}
