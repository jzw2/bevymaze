use bevy::utils::{HashMap, HashSet};
use bitvec::bitvec;
use bitvec::vec::BitVec;
use image::load;
use petgraph::graphmap::NodeTrait;
use petgraph::prelude::{GraphMap, UnGraphMap};
use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoEdges, IntoNeighbors};
use postcard::{from_bytes, to_stdvec};
use rand::rngs::{StdRng, ThreadRng};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::mem::size_of;
use std::ptr::null;
use std::sync::Mutex;

/// a node representing the "(r,theta)" coordinates of the maze
/// not really (r,theta), but "radius" and "sector"
/// all sectors are represented by inner-most clockwise-most corner in the sector
/// r is in [0, R] and theta in [0, N]
pub type CircleNode = (u64, i64);
/// a component, possibly a complete one
pub type MazeComponent<N> = UnGraphMap<N, bool>;

pub type CircleMazeComponent = MazeComponent<CircleNode>;

pub trait Maze<N: NodeTrait> {
    fn maze(&mut self) -> &mut MazeComponent<N>;
    fn set_maze(&mut self, maze: MazeComponent<N>);
    fn adjacent(&self, center: N) -> Vec<N>;
}

pub trait MazeBitRep {
    fn bit_rep(&self) -> BitVec<u32>;
}


pub struct CircleMaze {
    pub maze: CircleMazeComponent,

    /// the center in world-space
    pub center: (i64, i64),

    /// the radius in nodes
    ///
    /// represented as `R` in equations
    pub radius: u64,
    /// the minimum length of a cell wall, usually will be equal to the cell
    /// this will determine actual radius-units, and will determine
    /// the number of sectors in the inner-most ring
    ///
    /// in any equations we call this `s`
    pub cell_size: f64,

    /// the minimum width of a connection between cells
    /// should be smaller than cell_size, otherwise connections
    /// will be very rare
    ///
    /// represented as `p` in equations
    pub min_path_width: f64,

    /// the wall width, used when rendering the wall
    pub wall_width: f64,
}

impl CircleMaze {
    /// Get the nodes at a certain radius multiple
    /// actual eq is `floor(2 * pi * r * s / s)` but we simplify
    ///
    /// the rationale is that we're trying to pack as many
    /// cells that have an min side length of `s` as possible
    /// so we divide the total circumference (the numberator) by the side length
    /// this gives us how many side lengths exactly fit, but we floor since
    /// we can only have an integer number of nodes
    ///
    /// we call this `N` in most equations
    pub fn nodes_at_radius(&self, radius: u64) -> u64 {
        return (2.0 * PI * (radius as f64)).floor() as u64;
    }

    /// Get the actual angle of the node
    /// actual eq is `n/N * 2 * pi`
    pub fn angle_of_node(&self, node: CircleNode, nodes_at_radius: u64) -> f64 {
        return (node.1 as f64) / (nodes_at_radius as f64) * (2.0 * PI);
    }

    /// Get the angle that a single path width takes up at a certain radius
    /// actual eq is `p / (2*pi*r*s) * 2 * pi`
    pub fn path_angle(&self, at_radius: u64, min_path_width: f64) -> f64 {
        return min_path_width / (at_radius as f64 * self.cell_size);
    }

    /// Get the next node along the current ring
    /// modulo
    pub fn next_node(&self, node: CircleNode, increment: i64) -> CircleNode {
        return (node.0, node.1 + increment);
    }

    /// Map a possibly out of bounds node to it's in bounds counter-part
    pub fn correct_node(&self, node: CircleNode) -> CircleNode {
        if node.0 == 0 {
            return (0, 0);
        }
        return (
            node.0,
            node.1.rem_euclid(self.nodes_at_radius(node.0) as i64),
        );
    }

    /// Get all the nodes that share a wall of at least minimum_distance
    /// does NOT correct for negative/OOB angles
    pub fn touching(&self, center: CircleNode, minimum_distance: f64) -> Vec<CircleNode> {
        let mut nodes: Vec<CircleNode> = vec![];

        // we need to know how our node count compares to the others
        // called `N+k` or `N'`
        let above_count = self.nodes_at_radius(center.0 + 1);

        // center case
        // do this now because we have all the info we need
        if center.0 == 0 {
            // connect to everything around us
            for i in 0..above_count {
                nodes.push((1, i as i64));
            }
            return nodes;
        }

        // called `N`
        let our_count = self.nodes_at_radius(center.0);

        let our_next = self.next_node(center, 1);
        let our_prev = self.next_node(center, -1);
        // first add the obvious connections
        nodes.push(our_next);
        nodes.push(our_prev);

        // we will need this later
        let above_path_angle = self.path_angle(center.0 + 1, minimum_distance);
        let path_angle = self.path_angle(center.0, minimum_distance);
        let our_angle = self.angle_of_node(center, our_count);
        let our_next_angle = self.angle_of_node(our_next, our_count);

        if center.0 <= self.radius {
            // get the closest node in the ring above us
            // in equations we call this `n'`
            // eq is (r + 1, [theta / N * N'])
            let closest_above: CircleNode = (
                center.0 + 1,
                ((center.1 as f64) / (our_count as f64) * (above_count as f64)).round() as i64,
            );

            // full equation is
            // `ceil( (2*pi*(r+1)*s / (N + k)) / (2*pi*r/N) ) + 1`
            // but simplified
            // the rationale is that we want to know how many of the outer cells
            // will fit into the smaller ones
            let nodes_to_check_above = ((our_count as f64) * (center.0 as f64 + 1.0)
                / (above_count as f64 * center.0 as f64))
                .ceil() as i64
                + 1;

            // get all the nodes above us
            for i in 0..nodes_to_check_above {
                // the node we are currently checking
                let above_to_check = self.next_node(closest_above, i);

                let above_angle = self.angle_of_node(above_to_check, above_count);

                if i == 0 {
                    if above_angle > our_angle {
                        // if `theta_n < theta_n'` then we check the angle from `n'` to `n`
                        // eq is `theta_n' - theta_n > P_theta`
                        if above_angle - our_angle >= above_path_angle {
                            // this means we can fit a connection in BEFORE `n'`
                            nodes.push(self.next_node(above_to_check, -1));
                        }
                    }
                }

                // otherwise we simply check the distance to `theta_(n+1)`
                if our_next_angle - above_angle >= above_path_angle {
                    nodes.push(above_to_check);
                }
            }
        }

        if center.0 == 1 {
            // the center is the only one that connects here
            nodes.push((0, 0));
        } else {
            // lastly grab the nodes below us
            // called `N^`
            let below_count = self.nodes_at_radius(center.0 - 1);

            // get the closest node below
            // in equations we call this `n^`
            // eq is (r - 1, [theta / N * N^])
            let closest_below: CircleNode = (
                center.0 - 1,
                ((center.1 as f64) / (our_count as f64) * (below_count as f64)).round() as i64,
            );

            // full equation is
            // `ceil( (2*pi*(r-1)*s / (N^)) / (2*pi*r/N) ) + 1`
            // but simplified
            // the rationale is that we want to know how many of the outer cells
            // will fit into the smaller ones
            let nodes_to_check_below = ((our_count as f64) * (center.0 as f64 - 1.0)
                / (below_count as f64 * center.0 as f64))
                .ceil() as i64
                + 1;
            for i in 0..nodes_to_check_below {
                // the node we are currently checking
                let below_to_check = self.next_node(closest_below, i);

                let below_angle = self.angle_of_node(below_to_check, below_count);

                if i == 0 {
                    if our_angle < below_angle {
                        // we might be able to fit a connection before `theta_n^` aka `n^`
                        if below_angle - our_angle >= path_angle {
                            nodes.push(self.next_node(below_to_check, -1));
                        }
                    }
                }

                // in general check if we can fit a connection between `n^+i` and `n+1`
                if our_next_angle - below_angle >= path_angle {
                    nodes.push(below_to_check);
                }
            }
        }
        return nodes;
    }
}

impl Maze<CircleNode> for CircleMaze {
    fn maze(&mut self) -> &mut MazeComponent<CircleNode> {
        return &mut self.maze;
    }

    fn set_maze(&mut self, maze: MazeComponent<CircleNode>) {
        self.maze = maze;
    }

    /// center is (r, n)
    fn adjacent(&self, center: CircleNode) -> Vec<CircleNode> {
        return self
            .touching(center, self.min_path_width)
            .iter()
            .filter_map(|n| {
                return if n.0 <= self.radius {
                    Some(self.correct_node(*n))
                } else {
                    None
                };
            })
            .collect();
    }
}

fn next_i64_tuple(file: &mut File, mut buf: [u8; 8]) -> (i64, i64) {
    file.read_exact(&mut buf);
    let p1 = i64::from_be_bytes(buf);
    file.read_exact(&mut buf);
    let p2 = i64::from_be_bytes(buf);
    return (p1, p2);
}

/// This is an extension to the UnGraphMap that we use to store possible edges
/// when we are populating the maze
trait GetRandomNode<N: NodeTrait, E: Clone> {
    fn pop_random_edge(&mut self, neighbor_of: Option<N>) -> Option<(N, N)>;
    // TODO: remove the `weight` param, not sure how yet
    fn add_possible_edges_of_component<M: Maze<N>>(
        &mut self,
        maze: &M,
        component: &MazeComponent<N>,
        weight: E,
    );
    fn clean_edges_of_component(&mut self, component: &MazeComponent<N>);
}

impl<N: NodeTrait, E: Clone> GetRandomNode<N, E> for UnGraphMap<N, E> {
    fn pop_random_edge(&mut self, neighbor_of: Option<N>) -> Option<(N, N)> {
        if self.edge_count() == 0 {
            return None;
        }
        let rng = &mut thread_rng();
        let mut source = neighbor_of;
        let mut edge: Option<(N, N)> = None;
        if source.is_none() {
            // choose a source!
            source = self.nodes().choose(rng);
        } else {
            let neighbors = self.neighbors(source.unwrap());
            let neighbor = neighbors.choose(rng);
            if neighbor.is_none() {
                // we just have to redefine source
                source = self.nodes().choose(rng);
            } else {
                // we have all we need to set the edge immediately
                edge = Some((source.unwrap(), neighbor.unwrap()));
            }
        }

        // we haven't set edge so our source must be chosen by now
        if edge.is_none() {
            let src = source.unwrap();
            edge = Some((src, self.neighbors(src).choose(rng).unwrap()));
        }

        if let Some(e) = edge {
            self.remove_edge(e.0, e.1);
            if self.neighbors(e.0).count() == 0 {
                self.remove_node(e.0);
            }
            if self.neighbors(e.1).count() == 0 {
                self.remove_node(e.1);
            }

            return edge;
        }

        panic!("Could not choose an edge!");
    }

    // TODO: see about removing weight because it's kinda dumb
    // Adds all VALID possible edges to `self` that eminate from `component`.
    // A valid edge is an edge that does not connect any fully-connected components to themselves
    fn add_possible_edges_of_component<M: Maze<N>>(
        &mut self,
        maze: &M,
        component: &MazeComponent<N>,
        weight: E,
    ) {
        for node in component.nodes() {
            // get all edges of node that aren't between the component and itself
            for edge in
                maze.adjacent(node)
                    .into_iter()
                    .filter_map(|n| match component.contains_node(n) {
                        false => Some((node.clone(), n)),
                        true => None,
                    })
            {
                self.add_edge(edge.0, edge.1, weight.clone());
            }
        }
    }

    fn clean_edges_of_component(&mut self, component: &MazeComponent<N>) {
        // look into making this a lil faster?
        for edge in self.clone().all_edges() {
            if component.contains_node(edge.0) && component.contains_node(edge.1) {
                self.remove_edge(edge.0, edge.1);
                if self.neighbors(edge.0).count() == 0 {
                    self.remove_node(edge.0);
                }
                if self.neighbors(edge.1).count() == 0 {
                    self.remove_node(edge.1);
                }
            }
        }
    }
}

/// Generate a maze graph
/// The procedure is to connect and combine the components until we are left with a single
/// one. This single one will be the maze itself.
pub fn populate_maze<N: NodeTrait, M: Maze<N>>(
    graph: &mut M,
    mut starting_components: Vec<MazeComponent<N>>,
) -> &M
where
    M: Maze<N>,
{
    // generate a list of possible edges
    let mut possible_edges: UnGraphMap<N, bool> = UnGraphMap::new();
    for component in &starting_components {
        possible_edges.add_possible_edges_of_component(graph, component, true);
    }
    let mut last_sink: Option<N> = None;
    while let Some(mut new_edge) = possible_edges.pop_random_edge(last_sink) {
        // find the source and sink comps
        // the source comp needs to be a real component
        let source_comp_index = starting_components
            .iter_mut()
            .position(|c| c.contains_node(new_edge.0) || c.contains_node(new_edge.1))
            .unwrap();
        // swap the two because the rest of the code assumes that source_comp contains new_edge.0
        if !starting_components[source_comp_index].contains_node(new_edge.0) {
            new_edge = (new_edge.1, new_edge.0);
        }
        // first merge the two components (if the sink component exists)
        if let Some(sink_comp_index) = starting_components
            .iter()
            .position(|c| c.contains_node(new_edge.1))
            .clone()
        {
            let edges: Vec<_> = starting_components[sink_comp_index]
                .all_edges()
                .map(|(x, y, b)| (x, y, *b))
                .collect();
            // merge the components, ignoring edgeless nodes
            for edge in edges {
                starting_components[source_comp_index].add_edge(edge.0, edge.1, edge.2);
            }
            // now remove the sink component
            starting_components.remove(sink_comp_index);
        }
        // now we add the edge and check for new edges
        let source_comp_index = starting_components
            .iter_mut()
            .position(|c| c.contains_node(new_edge.0))
            .unwrap();
        let source_comp = &mut starting_components[source_comp_index];

        // we add the edge
        source_comp.add_edge(new_edge.0, new_edge.1, true);

        // finally update the possible edges
        // remove any possible edges that are no longer valid
        possible_edges.clean_edges_of_component(source_comp);
        // now search for new possible edges
        // we only have to look at the component we modified
        possible_edges.add_possible_edges_of_component(graph, source_comp, true);

        // finally finally set the last sink so we can continue on and create a meandering path
        last_sink = Some(new_edge.1);
    }
    graph.set_maze(starting_components.pop().unwrap());
    return graph;
}
