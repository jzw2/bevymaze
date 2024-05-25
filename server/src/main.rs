use bevy::prelude::Color;
use bitvec::vec::BitVec;
use futures_util::{SinkExt, StreamExt};
use log::*;
use postcard::{from_bytes, to_stdvec};
use server::connection::{CompressedMazeComponent, MazeNetworkRequest, MazeNetworkResponse, TerrainDataPoint};
use server::maze_gen::{
    add_next_edge, init_possible_edges, populate_maze, GetRandomNode, Maze, MazeBitRep,
    MazeComponent,
};
use server::square_maze_gen::{
    load_or_generate_component, square_starting_nodes, SquareMaze, SquareMazeComponent, SquareNode,
    SQUARE_MAZE_CELL_COUNT,
};
use server::terrain_gen::TerrainGenerator;
use std::hash::Hash;
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message::Binary;
use tokio_tungstenite::{
    accept_async,
    tungstenite::{Error, Result},
};

async fn accept_connection(peer: SocketAddr, stream: TcpStream) {
    if let Err(e) = handle_connection(peer, stream).await {
        match e {
            Error::ConnectionClosed | Error::Protocol(_) | Error::Utf8 => (),
            err => error!("Error processing connection: {}", err),
        }
    }
}

async fn handle_connection(peer: SocketAddr, stream: TcpStream) -> Result<()> {
    let mut ws_stream = accept_async(stream).await.expect("Failed to accept");

    let gen = TerrainGenerator::new();

    info!("New WebSocket connection: {}", peer);

    while let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        if let Binary(bin) = msg {
            // println!("Intercepted message!");
            let mut to_get = from_bytes::<MazeNetworkRequest>(&bin).unwrap();
            match to_get {
                MazeNetworkRequest::ReqTerrainHeights(mut data) => {
                    println!("RTH {}", data.len());
                    for point in &mut data {
                        // println!("BEFORE {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
                        let [x, y] = point.coordinates;
                        point.height = gen.get_height_for(x as f64, y as f64) as f32;
                        point.gradient = gen.get_gradient(x as f64, y as f64).as_vec2().to_array();
                        // println!("AFTER {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
                    }
                    ws_stream
                        .send(Binary(
                            to_stdvec(&MazeNetworkResponse::TerrainHeights(data)).unwrap(),
                        ))
                        .await?;
                }
                MazeNetworkRequest::ReqMaze(data) => {
                    println!("RM {}", data.len());
                    let filled: Vec<CompressedMazeComponent> = data
                        .iter()
                        .map(|cell| {
                            let [x, y] = cell;
                            return CompressedMazeComponent {
                                position: [*x, *y],
                                data: load_or_generate_component((*x as i64, *y as i64)).bit_rep(),
                            };
                        })
                        .collect();
                    for data in &filled {
                        println!("RM DATA {:?}", data.data.clone().into_vec());
                    }
                    ws_stream
                        .send(Binary(
                            to_stdvec(&MazeNetworkResponse::Maze(filled)).unwrap(),
                        ))
                        .await?;
                }
            }
            // println!("Fetching {}", to_get.len());
        }
    }

    Ok(())
}

use image::{ImageBuffer, Rgb, RgbImage};
use petgraph::graphmap::{NodeTrait, UnGraphMap};

const CELLS: i64 = 3;

const LEFT: i64 = -1;
const TOP: i64 = -1;

const WIDTH: u32 = (CELLS * SQUARE_MAZE_CELL_COUNT) as u32;
const HEIGHT: u32 = WIDTH;

const CELL_PIXELS: u32 = 8;
fn setup_maze_image() -> RgbImage {
    let width: u32 = (WIDTH + 2) * CELL_PIXELS;
    let height: u32 = (HEIGHT + 2) * CELL_PIXELS;

    let mut image: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));

    for i in 0..WIDTH + 2 {
        for j in 0..HEIGHT + 2 {
            for pi in 0..CELL_PIXELS / 2 {
                for pj in 0..CELL_PIXELS / 2 {
                    let pi = i * CELL_PIXELS + pi + CELL_PIXELS / 2;
                    let pj = j * CELL_PIXELS + pj + CELL_PIXELS / 2;
                    *image.get_pixel_mut(pi, pj) = Rgb([255, 255, 255]);
                }
            }
        }
    }
    return image;
}

fn floor_div(a: i64, b: i64) -> i64 {
    if a >= 0 {
        return a / b;
    }
    return (a - b + 1) / b;
}

fn draw_maze_from_bits(comps: Vec<Vec<BitVec<u32>>>, image: &mut RgbImage) {
    for s in -1..SQUARE_MAZE_CELL_COUNT * CELLS + 1 {
        for t in -1..SQUARE_MAZE_CELL_COUNT * CELLS + 1 {
            // first calculate the component
            let i = (s + LEFT * SQUARE_MAZE_CELL_COUNT).div_euclid(SQUARE_MAZE_CELL_COUNT) - LEFT;
            let j = (t + TOP * SQUARE_MAZE_CELL_COUNT).div_euclid(SQUARE_MAZE_CELL_COUNT) - TOP;
            if i < 0 || j < 0 {
                continue;
            }
            if i >= CELLS || j >= CELLS {
                continue;
            }
            let comp = &comps[i as usize][j as usize];
            let comp_pos = i * CELLS + j;
            let comp_color = Color::hsl(comp_pos as f32 * 20., 1.0, 0.5).as_rgba();
            let norm_color = (255. * comp_color.rgb_to_vec3()).round();
            let color = Rgb([norm_color.x as u8, norm_color.y as u8, norm_color.z as u8]);

            // now calculate the grid cell
            let x = (s + LEFT * SQUARE_MAZE_CELL_COUNT).rem_euclid(SQUARE_MAZE_CELL_COUNT);
            let y = (t + TOP * SQUARE_MAZE_CELL_COUNT).rem_euclid(SQUARE_MAZE_CELL_COUNT);
            let pos = 2 * (x + y * SQUARE_MAZE_CELL_COUNT) as usize;

            // check for the left edge
            let cpx = CELL_PIXELS * (s + 1) as u32;
            let cpy = CELL_PIXELS * (t + 1) as u32;
            if comp[pos] {
                for v in 0..CELL_PIXELS {
                    for w in 0..CELL_PIXELS / 4 {
                        let lp = (
                            cpx + v + CELL_PIXELS / 8 - CELL_PIXELS / 4,
                            cpy + w + CELL_PIXELS / 8 + CELL_PIXELS / 2,
                        );
                        *image.get_pixel_mut(lp.0, lp.1) = color;
                    }
                }
            }

            if comp[pos + 1] {
                for v in 0..CELL_PIXELS / 4 {
                    for w in 0..CELL_PIXELS {
                        let tp = (
                            cpx + v + CELL_PIXELS / 8 + CELL_PIXELS / 2,
                            cpy + w + CELL_PIXELS / 8 - CELL_PIXELS / 4,
                        );
                        *image.get_pixel_mut(tp.0, tp.1) = color;
                    }
                }
            }
        }
    }
}

fn draw_maze_graph(comps: Vec<&UnGraphMap<SquareNode, bool>>, image: &mut RgbImage) {
    for (idx, comp) in comps.into_iter().enumerate() {
        let comp_color = Color::hsl(idx as f32 * 20., 1.0, 0.5).as_rgba();
        let norm_color = (255. * comp_color.rgb_to_vec3()).round();
        let color = Rgb([norm_color.x as u8, norm_color.y as u8, norm_color.z as u8]);
        for s in -1..SQUARE_MAZE_CELL_COUNT * CELLS + 1 {
            for t in -1..SQUARE_MAZE_CELL_COUNT * CELLS + 1 {
                // check for the left edge
                let cpx = CELL_PIXELS * (s + 1) as u32;
                let cpy = CELL_PIXELS * (t + 1) as u32;
                let node = (
                    s + LEFT * SQUARE_MAZE_CELL_COUNT,
                    t + TOP * SQUARE_MAZE_CELL_COUNT,
                );
                if comp.contains_edge((node.0 - 1, node.1), node) {
                    for v in 0..CELL_PIXELS {
                        for w in 0..CELL_PIXELS / 4 {
                            let lp = (
                                cpx + v + CELL_PIXELS / 8 - CELL_PIXELS / 4,
                                cpy + w + CELL_PIXELS / 8 + CELL_PIXELS / 2,
                            );
                            *image.get_pixel_mut(lp.0, lp.1) = color;
                        }
                    }
                }

                if comp.contains_edge((node.0, node.1 - 1), node) {
                    for v in 0..CELL_PIXELS / 4 {
                        for w in 0..CELL_PIXELS {
                            let tp = (
                                cpx + v + CELL_PIXELS / 8 + CELL_PIXELS / 2,
                                cpy + w + CELL_PIXELS / 8 - CELL_PIXELS / 4,
                            );
                            *image.get_pixel_mut(tp.0, tp.1) = color;
                        }
                    }
                }

                if comp.contains_node(node) {
                    for v in 0..CELL_PIXELS / 4 {
                        for w in 0..CELL_PIXELS / 4 {
                            let tp = (
                                cpx + CELL_PIXELS / 8 + CELL_PIXELS / 2 + v,
                                cpy + CELL_PIXELS / 8 + CELL_PIXELS / 2 + w,
                            );
                            *image.get_pixel_mut(tp.0, tp.1) = color;
                        }
                    }
                }
            }
        }
    }
}

fn populate_maze_frame_by_frame(
    graph: &mut SquareMaze,
    mut starting_components: Vec<SquareMazeComponent>,
) -> &SquareMaze {
    // generate a list of possible edges
    let mut possible_edges = init_possible_edges(graph, &mut starting_components);
    let mut last_sink: Option<SquareNode> = None;

    // let mut maze_image = setup_maze_image();
    // let mut possible_edges_image = setup_maze_image();

    // let mut fc = 0;

    // draw_maze_graph(vec![&possible_edges], &mut possible_edges_image);
    // draw_maze_graph(starting_components.iter().collect(), &mut maze_image);
    //
    // maze_image
    //     .save(format!(
    //         "maze_img/maze_{}_{}.{fc}.png",
    //         graph.cell.0, graph.cell.1
    //     ))
    //     .expect("TODO: panic message");
    // possible_edges_image
    //     .save(format!(
    //         "maze_img/possible_{}_{}.{fc}.png",
    //         graph.cell.0, graph.cell.1
    //     ))
    //     .expect("TODO: panic message");

    while let Some(new_edge) = possible_edges.pop_random_edge(last_sink) {
        // fc += 1;

        add_next_edge(
            graph,
            &mut starting_components,
            new_edge,
            &mut possible_edges,
            &mut last_sink,
        );

        // let mut maze_image = setup_maze_image();
        // let mut possible_edges_image = setup_maze_image();
        //
        // draw_maze_graph(vec![&possible_edges], &mut possible_edges_image);
        // draw_maze_graph(starting_components.iter().collect(), &mut maze_image);
        //
        // maze_image
        //     .save(format!(
        //         "maze_img/maze_{}_{}.{fc}.png",
        //         graph.cell.0, graph.cell.1
        //     ))
        //     .expect("TODO: panic message");
        // possible_edges_image
        //     .save(format!(
        //         "maze_img/possible_{}_{}.{fc}.png",
        //         graph.cell.0, graph.cell.1
        //     ))
        //     .expect("TODO: panic message");
    }
    graph.set_maze(starting_components.pop().unwrap());
    return graph;
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // for i in 0..CELLS {
    //     for j in 0..CELLS {
    //         let cell = (i + LEFT, j + TOP);
    //         let mut maze = SquareMaze::new(cell);
    //         let starting = square_starting_nodes(&maze);
    //         populate_maze_frame_by_frame(&mut maze, starting);
    //         maze.save();
    //     }
    // }
    // 
    // let mut comps = vec![];
    // let mut bits = vec![];
    // 
    // for i in 0..CELLS {
    //     let mut col = vec![];
    //     for j in 0..CELLS {
    //         let cell = (i + LEFT, j + TOP);
    //         let mut maze = SquareMaze::load(cell).unwrap();
    //         {
    //             comps.push(maze.maze.clone());
    //         }
    //         col.push(maze.bit_rep());
    //     }
    //     bits.push(col);
    // }
    // 
    // let mut maze_image = setup_maze_image();
    // let mut bit_maze_image = setup_maze_image();
    // 
    // draw_maze_graph(comps.iter().collect(), &mut maze_image);
    // draw_maze_from_bits(bits, &mut bit_maze_image);
    // 
    // maze_image
    //     .save(format!("maze_img/maze_final.png"))
    //     .expect("TODO: panic message");
    // 
    // bit_maze_image
    //     .save(format!("maze_img/bit_maze_final.png"))
    //     .expect("TODO: panic message");

    let addr = "127.0.0.1:9002";
    let listener = TcpListener::bind(&addr).await.expect("Can't listen");
    info!("Listening on: {}", addr);
    
    while let Ok((stream, _)) = listener.accept().await {
        let peer = stream
            .peer_addr()
            .expect("connected streams should have a peer address");
        info!("Peer address: {}", peer);
    
        tokio::spawn(accept_connection(peer, stream));
    }
}
