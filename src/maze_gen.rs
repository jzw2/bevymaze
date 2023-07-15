use bevy::prelude::system_adapter::new;
use bevy::utils::{HashMap, HashSet};
use petgraph::graphmap::NodeTrait;
use petgraph::prelude::{GraphMap, UnGraphMap};
use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoEdges, IntoNeighbors};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{thread_rng, Rng};
use std::f64::consts::PI;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::ptr::null;

/// a node representing the x/y coordinates of the maze intersection
pub type SquareNode = (i64, i64);
/// a node representing the "(r,theta)" coordinates of the maze
/// not really (r,theta), but "radius" and "sector"
/// all sectors are represented by inner-most clockwise-most corner in the sector
/// r is in [0, R] and theta in [0, N]
pub type CircleNode = (u64, i64);
/// a component, possibly a complete one
pub type MazeComponent<N> = UnGraphMap<N, bool>;
pub type SquareMazeComponent = MazeComponent<SquareNode>;
pub type CircleMazeComponent = MazeComponent<CircleNode>;

pub trait Maze<N: NodeTrait> {
    fn maze(&mut self) -> &mut MazeComponent<N>;
    fn set_maze(&mut self, maze: MazeComponent<N>);
    fn adjacent(&self, center: N) -> Vec<N>;
}

/// Represent a square grid maze
pub struct SquareMaze {
    pub(crate) maze: SquareMazeComponent,
    pub(crate) size: i64,
    pub(crate) offset: (i64, i64),
}

impl Maze<SquareNode> for SquareMaze {
    fn maze(&mut self) -> &mut MazeComponent<SquareNode> {
        return &mut self.maze;
    }

    fn set_maze(&mut self, maze: MazeComponent<SquareNode>) {
        self.maze = maze;
    }

    fn adjacent(&self, center: SquareNode) -> Vec<SquareNode> {
        let mut nodes: Vec<SquareNode> = vec![];
        let size_off: SquareNode = (self.offset.0 + self.size, self.offset.1 + self.size);
        if center.0 + 1 < size_off.0 {
            nodes.push((center.0 + 1, center.1))
        }
        if center.1 + 1 < size_off.1 {
            nodes.push((center.0, center.1 + 1))
        }
        if center.0 - 1 >= self.offset.0 {
            nodes.push((center.0 - 1, center.1))
        }
        if center.1 - 1 >= self.offset.1 {
            nodes.push((center.0, center.1 - 1))
        }
        return nodes;
    }
}

pub struct CircleMaze {
    pub(crate) maze: CircleMazeComponent,

    /// the center in world-space
    pub(crate) center: (i64, i64),

    /// the radius in nodes
    ///
    /// represented as `R` in equations
    pub(crate) radius: u64,
    /// the minimum length of a cell wall, usually will be equal to the cell
    /// this will determine actual radius-units, and will determine
    /// the number of sectors in the inner-most ring
    ///
    /// in any equations we call this `s`
    pub(crate) cell_size: f64,

    /// the minimum width of a connection between cells
    /// should be smaller than cell_size, otherwise connections
    /// will be very rare
    ///
    /// represented as `p` in equations
    pub(crate) min_path_width: f64,
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
    fn nodes_at_radius(&self, radius: u64) -> u64 {
        return (2.0 * PI * (radius as f64)).floor() as u64;
    }

    /// Get the actual angle of the node
    /// actual eq is `n/N * 2 * pi * r * s`
    fn angle_of_node(&self, node: CircleNode, nodes_at_radius: u64) -> f64 {
        return (node.1 as f64) / (nodes_at_radius as f64)
            * (2.0 * PI * (node.0 as f64) * self.cell_size);
    }

    /// Get the angle that a single path width takes up at a certain radius
    /// actual eq is `p / (2*pi*r*s)`
    fn path_angle(&self, at_radius: u64) -> f64 {
        return self.min_path_width / (2.0 * PI * at_radius as f64 * self.cell_size);
    }

    /// Get the next node along the current ring
    /// modulo
    fn next_node(&self, node: CircleNode, increment: i64, nodes_at_radius: u64) -> CircleNode {
        return (
            node.0,
            (node.1 + increment).rem_euclid(nodes_at_radius as i64),
        );
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

        let our_next = self.next_node(center, 1, our_count);
        let our_prev = self.next_node(center, -1, our_count);
        // first add the obvious connections
        nodes.push(our_next);
        nodes.push(our_prev);

        // get the closest node in the ring above us
        // in equations we call this `n'`
        // eq is (r + 1, [theta / N * N'])
        let closest_above: CircleNode = (
            center.0 + 1,
            ((center.1 as f64) / (our_count as f64) * (above_count as f64)).round() as i64,
        );

        // we will need this later
        let above_path_angle = self.path_angle(center.0 + 1);
        let path_angle = self.path_angle(center.0);
        let our_angle = self.angle_of_node(center, our_count);
        let our_next_angle = self.angle_of_node(our_next, our_count);

        if center.0 <= self.radius {
            // full equation is
            // `ceil( (2*pi*(r+1)*s / (N + k)) / (2*pi*r/N) )`
            // but simplified
            // the rationale is that we want to know how many of the outer cells
            // will fit into the smaller ones
            let nodes_to_check_above = ((our_count as f64) * (center.0 as f64 + 1.0)
                / (above_count as f64 * center.0 as f64))
                .ceil() as i64;
            // get all the nodes above us
            for i in 0..nodes_to_check_above {
                // the node we are currently checking
                let above_to_check = self.next_node(closest_above, i, above_count);

                let above_angle = self.angle_of_node(above_to_check, above_count);

                if i == 0 {
                    // when i = 0, we need to use `n` as the distance measure, not `n+1`
                    if above_angle < our_angle {
                        // if `theta_n < theta_n'` then we check the angle
                        // from `n'` to `n`
                        // eq is `theta_n' - theta_n > P_theta`
                        if above_angle - our_angle >= above_path_angle {
                            // this means we can fit a connection in BEFORE `n'`
                            nodes.push(self.next_node(above_to_check, -1, above_count));
                        }
                    } else if above_angle > our_angle {
                        // if `theta_n > theta_n'` then we check the distance/angle
                        // from `n` to `n'+1` to see if we can fit a node
                        // equation is `theta_(n'+1) - theta_n > P_theta`
                        // where P_theta is the min path width, will be `s` here
                        if self.angle_of_node(
                            self.next_node(above_to_check, 1, above_count),
                            above_count,
                        ) - our_angle
                            >= above_path_angle
                        {
                            // this means we can fit a connection in AFTER `n`
                            nodes.push(above_to_check);
                        }
                    } else {
                        if our_next_angle - above_angle >= above_path_angle {
                            nodes.push(above_to_check);
                        }
                    }
                } else {
                    // otherwise we simply check the distance to `theta_(n+1)`
                    if our_next_angle - above_angle >= above_path_angle {
                        nodes.push(above_to_check);
                    }
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
                ((center.1 as f64) * ((above_count as f64) / (our_count as f64))).round() as i64,
            );

            // full equation is
            // `ceil( (2*pi*(r-1)*s / (N^)) / (2*pi*r/N) )`
            // but simplified
            // the rationale is that we want to know how many of the outer cells
            // will fit into the smaller ones
            let nodes_to_check_below = ((our_count as f64) * (center.0 as f64 - 1.0)
                / (below_count as f64 * center.0 as f64))
                .ceil() as i64;
            for i in 0..nodes_to_check_below {
                // the node we are currently checking
                let below_to_check = self.next_node(closest_below, i, below_count);

                let below_angle = self.angle_of_node(below_to_check, below_count);

                if i == 0 {
                    if our_angle < below_angle {
                        // we might be able to fit a connection before `theta_n^` aka `n^`
                        if below_angle - our_angle >= path_angle {
                            nodes.push(self.next_node(below_to_check, -1, below_count));
                        }
                    } else if our_angle > below_angle {
                        //check if we fit a connection between `n^+1` and `n`
                        if self.angle_of_node(
                            self.next_node(below_to_check, 1, below_count),
                            below_count,
                        ) - our_angle
                            >= path_angle
                        {
                            nodes.push(below_to_check)
                        }
                    } else {
                        //check if we fit a connection between `n^` and `n+1`
                        if our_next_angle - below_angle >= path_angle {
                            nodes.push(below_to_check);
                        }
                    }
                } else {
                    // default case is to check if we fit a connection between `n^+i` and `n+1`
                    if our_next_angle - below_angle >= path_angle {
                        nodes.push(below_to_check);
                    }
                }
            }
        }
        return nodes;
    }
}

/// Divide the "n"umerator from the "d"enominator and round up
fn ceil_div(n: i64, d: i64) -> i64 {
    if n % d == 0 {
        return n / d;
    } else {
        return n / d + 1;
    }
}

fn next_i64_tuple(file: &mut File, mut buf: [u8; 8]) -> (i64, i64) {
    file.read_exact(&mut buf);
    let p1 = i64::from_be_bytes(buf);
    file.read_exact(&mut buf);
    let p2 = i64::from_be_bytes(buf);
    return (p1, p2);
}

impl SquareMaze {
    fn save(&self) {
        self.save_named(SquareMaze::file_name(self.offset).as_str());
    }

    fn save_named(&self, fname: &str) {
        let file_res = OpenOptions::new()
            .append(true) // Open in append mode
            .open(fname);
        let mut file = file_res.unwrap();
        file.write(&self.size.to_be_bytes());
        file.write(&self.offset.0.to_be_bytes());
        file.write(&self.offset.1.to_be_bytes());
        for x in self.offset.0 - 1..self.offset.0 + self.size + 1 {
            for y in self.offset.1 - 1..self.offset.1 + self.size + 1 {
                let n1 = (x, y);
                let n2 = (x + 1, y);
                if self.maze.contains_edge(n1, n2) {
                    file.write(&n1.0.to_be_bytes());
                    file.write(&n1.1.to_be_bytes());
                    file.write(&n2.0.to_be_bytes());
                    file.write(&n2.1.to_be_bytes());
                }
            }
        }
        file.flush();
    }

    fn file_name(offset: (i64, i64)) -> String {
        return offset.0.to_string() + "," + &*offset.1.to_string() + ".mzb";
    }

    fn load(offset: (i64, i64)) -> Self {
        return SquareMaze::load_named(SquareMaze::file_name(offset).as_str());
    }

    fn load_named(fname: &str) -> Self {
        let mut ret = SquareMaze {
            maze: UnGraphMap::new(),
            size: 0,
            offset: (0, 0),
        };
        let file_res = File::open(fname);
        let mut file = file_res.unwrap();
        let mut buf_i64: [u8; 8] = [0; 8];
        file.read_exact(&mut buf_i64);
        ret.size = i64::from_be_bytes(buf_i64);
        ret.offset = next_i64_tuple(&mut file, buf_i64);
        // the size of the 'size' and 'offset' stuff at the beginning of the file
        const PREAMBLE_SIZE: u64 = 8 + 2 * 8;
        // the size of an edge in bytes; rep'd in the file as 2 i64 tuples
        const EDGE_SIZE: u64 = 8 * 4;
        let edge_count = (file.metadata().unwrap().len() - PREAMBLE_SIZE) / EDGE_SIZE;
        for _ in 0..edge_count {
            let source = next_i64_tuple(&mut file, buf_i64);
            let sink = next_i64_tuple(&mut file, buf_i64);
            ret.maze.add_edge(source, sink, true);
        }
        return ret;
    }
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

const MAZE_CHUNK_SIZE: u16 = 32;

fn starting_nodes(graph: SquareMaze) {
    let left = SquareMaze::load((graph.offset.0 - graph.size, graph.offset.1));
    let right = SquareMaze::load((graph.offset.0 + graph.size, graph.offset.1));
    let top = SquareMaze::load((graph.offset.0, graph.offset.1 - graph.size));
    let bottom = SquareMaze::load((graph.offset.0, graph.offset.1 + graph.size));
}

/// Generate a maze graph
/// The procedure is to connect connect and combine the components until we are left with a single
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
