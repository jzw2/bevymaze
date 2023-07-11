use bevy::utils::HashSet;
use petgraph::prelude::{GraphMap, UnGraphMap};
use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoEdges};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

/// utility functions
/// copied from https://stackoverflow.com/a/53773240/3210986
struct VecSet<T> {
    set: HashSet<T>,
    vec: Vec<T>,
}

impl<T> VecSet<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    fn new() -> Self {
        Self {
            set: HashSet::new(),
            vec: Vec::new(),
        }
    }
    fn insert(&mut self, elem: T) {
        assert_eq!(self.set.len(), self.vec.len());
        let was_new = self.set.insert(elem.clone());
        if was_new {
            self.vec.push(elem);
        }
    }
    fn remove_random(&mut self) -> T {
        assert_eq!(self.set.len(), self.vec.len());
        let index = thread_rng().gen_range(0..self.vec.len());
        let elem = self.vec.swap_remove(index);
        let was_present = self.set.remove(&elem);
        assert!(was_present);
        elem
    }
    fn is_empty(&self) -> bool {
        assert_eq!(self.set.len(), self.vec.len());
        self.vec.is_empty()
    }

    /// end stackoverflow section

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.vec.retain_mut(|el| {
            if f(el) {
                return true;
            }
            self.set.remove(el);
            false
        });
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
        nodes.shuffle(&mut thread_rng());
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

fn add_possible_edges_of_component(
    graph: &SquareMaze,
    possible_edges: &mut VecSet<((i64, i64), (i64, i64))>,
    component: &SquareMazeComponent,
) {
    for node in component.nodes() {
        // get all edges of node that aren't between the component and itself
        for node in
            graph
                .adjacent(node)
                .into_iter()
                .filter_map(|n| match component.contains_node(n) {
                    false => Some((node.clone(), n)),
                    true => None,
                })
        {
            possible_edges.insert(node);
        }
    }
}

/// Generate a maze graph
/// The procedure is to connect connect and combine the components until we are left with a single
/// one. This single one will be the maze itself.
pub fn populate_maze(
    graph: &mut SquareMaze,
    mut starting_components: Vec<SquareMazeComponent>,
) -> &SquareMaze {
    // generate a list of possible edges
    let mut possible_edges: VecSet<(SquareNode, SquareNode)> = VecSet::new();
    for component in &starting_components {
        add_possible_edges_of_component(&graph, &mut possible_edges, component);
    }
    // let last_sink: SquareNode;
    while !possible_edges.is_empty() {
        let new_edge = possible_edges.remove_random();
        let source_comp_index = starting_components
            .iter_mut()
            .position(|c| c.contains_node(new_edge.0))
            .unwrap();
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
        possible_edges
            .retain(|e| !(source_comp.contains_node(e.0) && source_comp.contains_node(e.1)));
        // now search for new possible edges
        // we only have to look at the component we modified
        add_possible_edges_of_component(&graph, &mut possible_edges, source_comp);
    }
    graph.maze = starting_components.pop().unwrap();
    return graph;
}
