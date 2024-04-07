use crate::maze_gen::{populate_maze, CompressedMaze, Maze, MazeComponent};
use bitvec::bitvec;
use bitvec::prelude::BitVec;
use postcard::{from_bytes, to_stdvec};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;

pub const SQUARE_MAZE_CELL_COUNT: i64 = 16;
pub const SQUARE_MAZE_CELL_SIZE: f64 = 2.0f64;
pub const SQUARE_MAZE_WALL_WIDTH: f64 = 0.05f64;

/// a node representing the x/y coordinates of the maze intersection
pub type SquareNode = (i64, i64);

pub type SquareMazeComponent = MazeComponent<SquareNode>;

impl SquareMaze {
    pub fn is_outside_maze(&self, node: SquareNode, fuzz: i64) -> bool {
        let (x, y) = self.offset();
        let (i, j) = node;
        return i < x - fuzz
            || j < y - fuzz
            || i >= x + self.size + fuzz
            || j > y + self.size + fuzz;
    }

    pub fn new(cell: (i64, i64)) -> Self {
        return SquareMaze {
            maze: Default::default(),
            size: SQUARE_MAZE_CELL_COUNT,
            cell,
            cell_size: SQUARE_MAZE_CELL_SIZE,
            wall_width: SQUARE_MAZE_WALL_WIDTH,
        };
    }

    pub fn offset(&self) -> (i64, i64) {
        return (self.cell.0 * self.size, self.cell.1 * self.size);
    }

    pub fn bit_rep(&self) -> CompressedSquareMaze {
        let mut bits = bitvec![0; (self.size*self.size) as usize*2];
        for x in 0..self.size {
            for y in 0..self.size {
                let node = (x + self.offset().0, y + self.offset().1);
                let pos = 2 * (x + y * self.size) as usize;
                if self.maze.contains_edge(node, (node.0 - 1, node.1)) {
                    bits.set(pos, true);
                }
                if self.maze.contains_edge(node, (node.0, node.1 - 1)) {
                    bits.set(pos + 1, true);
                }
            }
        }
        return CompressedSquareMaze {
            edges: bits,
            size: self.size,
            cell: self.cell,
        };
    }

    pub fn save(&self) {
        self.save_named(SquareMaze::file_name(self.cell).as_str());
    }

    fn save_named(&self, fname: &str) {
        let mut to_save: SquareMaze = self.clone();
        for node in self.maze.nodes() {
            if self.is_outside_maze(node, 1) {
                to_save.maze.remove_node(node);
            }
        }
        let file_res = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            // .append(true) // Open in append mode
            .open(fname);

        let obj = to_stdvec::<SquareMaze>(&to_save).expect("Could not serialize maze");
        let mut file = file_res.expect("Could not open file");
        file.write(&obj).expect("Could not write maze");
        file.flush().expect("Could not flush maze file");
    }

    fn file_name(cell: (i64, i64)) -> String {
        return cell.0.to_string() + "_" + &*cell.1.to_string() + ".mzb";
    }

    pub fn load(cell: (i64, i64)) -> Option<Self> {
        return SquareMaze::load_named(SquareMaze::file_name(cell).as_str());
    }

    fn load_named(fname: &str) -> Option<Self> {
        let file_res = fs::read(fname);

        if file_res.is_err() {
            return None;
        }

        let ret = from_bytes::<SquareMaze>(&file_res.unwrap());

        if ret.is_err() {
            return None;
        }

        return Some(ret.unwrap());
    }
}

pub struct CompressedSquareMaze {
    /// The maze graph is assumed to have a node at every
    /// integer tuple `[offset_x, offset_x + size) x [offset_y, offset_y + size)`.
    /// This means we only store the edges. Furthermore, edges can only be between neighboring
    /// cells, so each cell only stores the left and top neighbors as booleans.
    /// For a cell `(x,y)` the left, top, right, and bottom edges exist based on the values of
    /// `edges[2*(x + y*size)], edges[2*(x + y*size) + 1], edges[2*(x + 1 + y*size)], edges[2*(x + (y+1)*size) + 1]`
    /// respectively
    pub edges: BitVec,
    /// the amount of cells on one side
    pub size: i64,
    /// the offset of nodes
    pub cell: (i64, i64),
}

/// Represent a square grid maze
#[derive(Serialize, Deserialize, Clone)]
pub struct SquareMaze {
    pub maze: SquareMazeComponent,
    /// the amount of cells on one side
    pub size: i64,
    /// the offset of nodes
    pub cell: (i64, i64),
    /// the width and height of each square
    pub cell_size: f64,
    /// the width of the walls
    pub wall_width: f64,
}
impl Maze<SquareNode> for SquareMaze {
    fn maze(&mut self) -> &mut MazeComponent<SquareNode> {
        return &mut self.maze;
    }

    fn set_maze(&mut self, maze: MazeComponent<SquareNode>) {
        self.maze = maze;
    }

    fn adjacent(&self, center: SquareNode) -> Vec<SquareNode> {
        let (x, y) = center;
        let mut nodes: Vec<SquareNode> = vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)];
        return nodes
            .iter()
            .filter_map(|n| match self.is_outside_maze(*n, 0) {
                true => None,
                false => Some(*n)
            })
            .collect();

        // // current bounds of maze component
        // let (x_o, y_o) = self.offset();
        // let size_off: SquareNode = (x_o + self.size, y_o + self.size);
        // for i in [-1, 1] {
        //     let n_x = center.0 + i;
        //     if x_o - 1 <= n_x && n_x <= size_off.0 {
        //         if y_o <= center.1 && center.1 < size_off.1 {
        //             nodes.push((n_x, center.1));
        //         }
        //     };
        // }

        // for i in [-1, 1] {
        //     let n_y = center.1 + i;
        //     if x_o <= center.1 && center.1 < size_off.0 {
        //         if y_o - 1 <= n_y && n_y <= size_off.1 {
        //             nodes.push((center.0, n_y));
        //         }
        //     };
        // }

        // return nodes;
    }
}

impl CompressedMaze for SquareMaze {
    fn compressed(&self) -> BitVec<u32> {
        let mut vec = BitVec::new();
        for x in 0..self.size {
            for y in 0..self.size {
                vec.push(self.maze.contains_edge((x - 1, y), (x, y)));
                vec.push(self.maze.contains_edge((x, y - 1), (x, y)));
            }
        }
        return vec;
    }
}

/// Adds starting nodes to `graph`
fn square_starting_nodes(graph: &SquareMaze) -> Vec<SquareMazeComponent> {
    let mut components = vec![];
    let mut rng = thread_rng();
    let (x, y) = graph.offset();

    if let Some(left) = SquareMaze::load((graph.cell.0 - 1, graph.cell.1)) {
        components.push(left.maze);
    } else {
        let mut comp = SquareMazeComponent::default();
        comp.add_node((x - 1, y + rng.gen_range(0, graph.size)));
        components.push(comp);
    }
    if let Some(right) = SquareMaze::load((graph.cell.0 + 1, graph.cell.1)) {
        components.push(right.maze);
    } else {
        let mut comp = SquareMazeComponent::default();
        comp.add_node((x + graph.size, y + rng.gen_range(0, graph.size)));
        components.push(comp);
    }

    if let Some(top) = SquareMaze::load((graph.cell.0, graph.cell.1 - 1)) {
        components.push(top.maze);
    } else {
        let mut comp = SquareMazeComponent::default();
        comp.add_node((x + rng.gen_range(0, graph.size), y - 1));
        components.push(comp);
    }

    if let Some(bottom) = SquareMaze::load((graph.cell.0, graph.cell.1 + 1)) {
        components.push(bottom.maze);
    } else {
        let mut comp = SquareMazeComponent::default();
        comp.add_node((x + rng.gen_range(0, graph.size), y + graph.size));
        components.push(comp);
    }

    return components;
}

pub fn load_or_generate_component(cell: (i64, i64)) -> SquareMaze {
    if let Some(loaded) = SquareMaze::load(cell) {
        return loaded;
    }

    let mut maze = SquareMaze::new(cell);
    let starting = square_starting_nodes(&maze);
    populate_maze(&mut maze, starting);

    maze.save();
    return maze;
}
