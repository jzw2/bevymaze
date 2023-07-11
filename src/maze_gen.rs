use petgraph::prelude::{GraphMap, UnGraphMap};
use petgraph::visit::{GetAdjacencyMatrix, IntoEdgeReferences, IntoEdges};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

const MAZE_CHUNK_SIZE: u16 = 32;

/// a node representing the x/y coordinates of the maze intersection
type SquareNode = (i64, i64);
/// a component, possibly a complete one
type SquareMazeComponent = UnGraphMap<SquareNode, bool>;

/// Represent a square grid maze
struct SquareMaze {
    maze: SquareMazeComponent,
    size: i64,
    offset: (i64, i64),
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

fn add_possible_edges_of_components(
    graph: &SquareMaze,
    possible_edges: &mut Vec<((i64, i64), (i64, i64))>,
    component: &SquareMazeComponent,
) {
    for node in component.nodes() {
        // get all edges of node that aren't between the component and itself
        possible_edges.extend(graph.adjacent(node).into_iter().filter_map(|n| {
            match component.contains_node(n) {
                true => Some((node.clone(), n)),
                false => None,
            }
        }));
    }
}

/// Generate a maze graph
/// The procedure is to connect connect and combine the components until we are left with a single
/// one. This single one will be the maze itself.
fn populate_maze(
    mut graph: SquareMaze,
    mut starting_components: Vec<SquareMazeComponent>,
) -> SquareMaze {
    // generate a list of possible edges
    let mut possible_edges: Vec<(SquareNode, SquareNode)> = Vec::new();
    for component in &starting_components {
        add_possible_edges_of_components(&graph, &mut possible_edges, component);
    }
    while let Some(new_edge) = possible_edges.pop() {
        // add the edge
        let source_comp_index = starting_components
            .iter_mut()
            .position(|c| c.contains_node(new_edge.0))
            .unwrap();
        starting_components[source_comp_index].add_edge(new_edge.0, new_edge.1, true);
        // now merge the two components
        if let Some(p) = starting_components
            .iter()
            .position(|c| c.contains_node(new_edge.0))
            .clone()
        {
            let edges: Vec<_> = starting_components[p]
                .all_edges()
                .map(|(x, y, b)| (x, y, *b))
                .collect();
            // now remove the sink component
            starting_components.remove(p);
            let source_comp = &mut starting_components[source_comp_index];
            // merge the components, ignoring edgeless nodes
            for edge in edges {
                source_comp.add_edge(edge.0, edge.1, edge.2);
            }

            // finally update the possible edges
            // remove any possible edges that are no longer valid
            possible_edges
                .retain(|e| !(source_comp.contains_node(e.0) && source_comp.contains_node(e.1)));
            // now add our new edges
            add_possible_edges_of_components(&graph, &mut possible_edges, source_comp);
        }
    }
    graph.maze = starting_components.pop().unwrap();
    return graph;
}
