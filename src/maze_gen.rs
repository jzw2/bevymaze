use bevy::prelude::system_adapter::new;
use bevy::utils::{HashMap, HashSet};
use petgraph::graphmap::NodeTrait;
use petgraph::prelude::{GraphMap, UnGraphMap};
use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoEdges, IntoNeighbors};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::{thread_rng, Rng};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::ptr::null;

/// utility object for storing possible edges
/// edges are stored as

trait GetRandomNode<N, E> {
    fn pop_random_edge(&mut self, neighbor_of: Option<N>) -> Option<(N, N)>;
    // TODO: re-add these to the trait once we properly generify the maze generation code
    // fn add_possible_edges_of_component(&mut self, maze: SquareMaze, component: SquareMazeComponent);
    // fn clean_edges_of_component(&mut self, component: SquareMazeComponent);
}

impl<N, E> GetRandomNode<N, E> for UnGraphMap<N, E>
where
    N: NodeTrait,
{
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
}

fn add_possible_edges_of_component2(
    possible_edges: &mut UnGraphMap<SquareNode, bool>,
    maze: &SquareMaze,
    component: &SquareMazeComponent,
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
            possible_edges.add_edge(edge.0, edge.1, true);
        }
    }
}

fn clean_edges_of_component(
    possible_edges: &mut UnGraphMap<SquareNode, bool>,
    component: &SquareMazeComponent,
) {
    // look into making this a lil faster?
    for edge in possible_edges.clone().all_edges() {
        if component.contains_node(edge.0) && component.contains_node(edge.1) {
            possible_edges.remove_edge(edge.0, edge.1);
            if possible_edges.neighbors(edge.0).count() == 0 {
                possible_edges.remove_node(edge.0);
            }
            if possible_edges.neighbors(edge.1).count() == 0 {
                possible_edges.remove_node(edge.1);
            }
        }
    }
}

const MAZE_CHUNK_SIZE: u16 = 32;

/// a node representing the x/y coordinates of the maze intersection
pub type SquareNode = (i64, i64);
/// a component, possibly a complete one
pub type SquareMazeComponent = UnGraphMap<SquareNode, bool>;

/// Represent a square grid maze
pub struct SquareMaze {
    pub(crate) maze: SquareMazeComponent,
    pub(crate) size: i64,
    pub(crate) offset: (i64, i64),
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
    fn adjacent(&self, center: (i64, i64)) -> Vec<(i64, i64)> {
        let mut nodes: Vec<(i64, i64)> = vec![];
        let size_off: (i64, i64) = (self.offset.0 + self.size, self.offset.1 + self.size);
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

fn starting_nodes(graph: SquareMaze) {
    let left = SquareMaze::load((graph.offset.0 - graph.size, graph.offset.1));
    let right = SquareMaze::load((graph.offset.0 + graph.size, graph.offset.1));
    let top = SquareMaze::load((graph.offset.0, graph.offset.1 - graph.size));
    let bottom = SquareMaze::load((graph.offset.0, graph.offset.1 + graph.size));
}

/// Generate a maze graph
/// The procedure is to connect connect and combine the components until we are left with a single
/// one. This single one will be the maze itself.
pub fn populate_maze(
    graph: &mut SquareMaze,
    mut starting_components: Vec<SquareMazeComponent>,
) -> &SquareMaze {
    // generate a list of possible edges
    let mut possible_edges: UnGraphMap<SquareNode, bool> = UnGraphMap::new();
    for component in &starting_components {
        add_possible_edges_of_component2(&mut possible_edges, &graph, component);
    }
    let mut last_sink: Option<SquareNode> = None;
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
        clean_edges_of_component(&mut possible_edges, source_comp);
        // now search for new possible edges
        // we only have to look at the component we modified
        add_possible_edges_of_component2(&mut possible_edges, &graph, source_comp);

        // finally finally set the last sink so we can continue on and create a meandering path
        last_sink = Some(new_edge.1);
    }
    graph.maze = starting_components.pop().unwrap();
    return graph;
}
